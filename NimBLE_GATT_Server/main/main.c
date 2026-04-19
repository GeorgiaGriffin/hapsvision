#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_log.h"
#include "esp_check.h"
#include "nvs_flash.h"

#include "driver/ledc.h"
#include "driver/i2c_master.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/util/util.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "store/config/ble_store_config.h"
#include "esp_timer.h"
#include "model_wrapper.h"


static const char *TAG = "BLE_SERVO";
 
/* ---------------- Servo settings ---------------- */

#define SERVO_COUNT          4
#define SERVO_FREQ_HZ        50
#define SERVO_PERIOD_US      20000
#define SERVO_MIN_PULSE_US   500
#define SERVO_MAX_PULSE_US   2500

#define SERVO_IDLE_DEG       90
#define SERVO_ACTIVE_DEG     135
#define SERVO_PULSE_MS       250
#define SERVO_PULSES         2

/* continuous "buzz" pattern during calibration */
#define SERVO_BUZZ_LOW_DEG   75
#define SERVO_BUZZ_HIGH_DEG  105
#define SERVO_BUZZ_MS        90

#define SERVO_LEDC_MODE      LEDC_LOW_SPEED_MODE
#define SERVO_LEDC_TIMER     LEDC_TIMER_0
#define SERVO_LEDC_RES       LEDC_TIMER_13_BIT

#define FORWARD_AFTER_TURN_ENABLE      1
#define FORWARD_AFTER_TURN_DELAY_MS    120
#define FORWARD_AFTER_TURN_COUNT       2

#define MPU_REG_ACCEL_CONFIG         0x1C
#define MPU_REG_ACCEL_XOUT_H         0x3B
#define MPU_REG_GYRO_XOUT_H          0x43

#define LOG_WINDOW_MS 3000

#define GESTURE_WINDOW    100
#define GESTURE_AXES      6
#define GESTURE_STRIDE_MS 250

#define TAP_MARGIN          8        // tap logit must beat idle by this much
#define TAP_COOLDOWN_MS     1200     // ignore taps within this window after a detection
#define MOTION_FLOOR        50000000.0f      // sum-of-squared-deltas; tune using logs


static const int servo_gpio[SERVO_COUNT] = {4, 5, 6, 7};

static const ledc_channel_t servo_channel[SERVO_COUNT] = {
    LEDC_CHANNEL_0,
    LEDC_CHANNEL_1,
    LEDC_CHANNEL_2,
    LEDC_CHANNEL_3
};

typedef enum {
    CMD_LEFT = 0,
    CMD_RIGHT,
    CMD_FORWARD,
    CMD_BACKWARD
} servo_cmd_t;

typedef struct {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
} mpu_sample_t;

typedef enum {
    LOG_LABEL_NONE = 0,
    LOG_LABEL_IDLE = 1,
    LOG_LABEL_TAP = 2,
    LOG_LABEL_FLICK = 3,
    LOG_LABEL_TURN = 4
} log_label_t;

static volatile bool g_log_enabled = false;
static int64_t g_log_start_ms = 0;
static volatile int  g_log_label = LOG_LABEL_IDLE;
void ble_store_config_init(void);

static const float NORM_MEAN[6] = { -1465.071045f, 9708.753906f, 5880.393555f, -183.500336f, -37.771332f, -360.437653f };
static const float NORM_STD[6]  = { 4485.135254f, 7815.969727f, 6202.984863f, 1734.409058f, 2344.524170f, 2519.624268f };

static int16_t g_ring[GESTURE_WINDOW][GESTURE_AXES];
static volatile int  g_ring_head = 0;
static volatile bool g_ring_full = false;
static portMUX_TYPE  g_ring_mux  = portMUX_INITIALIZER_UNLOCKED;

static volatile servo_cmd_t g_last_route_cmd = CMD_FORWARD;
static volatile bool        g_have_last_cmd  = false;

static int64_t last_tap_fire_ms = 0;

/* ---------------- MPU-6050 / I2C settings ---------------- */

#define MPU_SDA_GPIO                 8
#define MPU_SCL_GPIO                 9
#define MPU_I2C_PORT                 I2C_NUM_0
#define MPU_ADDR                     0x68  

#define MPU_REG_GYRO_CONFIG          0x1B
#define MPU_REG_GYRO_ZOUT_H          0x47
#define MPU_REG_PWR_MGMT_1           0x6B
#define MPU_REG_WHO_AM_I             0x75

/* We set gyro full-scale to +-500 dps => 65.5 LSB/(deg/s) */
#define MPU_GYRO_SENS_LSB_PER_DPS    65.5f

#define MPU_SAMPLE_MS                10
#define MPU_BIAS_SAMPLES             200
#define CAL_TOL_DEG                  12.0f

/*
 * IMPORTANT:
 * Depending on how the MPU-6050 is mounted, left turns may produce
 * positive or negative Z gyro. If LEFT never completes but RIGHT does,
 * change this from +1.0f to -1.0f.
 */

#define MPU_YAW_SIGN                 (-1.0f)

/* ---------------- Calibration state ---------------- */

typedef enum {
    NAV_WAITING_CALIBRATE = 0,
    NAV_WAITING_TURN_TARGET,
    NAV_CALIBRATING,
    NAV_ROUTING
} nav_state_t;

typedef enum {
    BLE_MSG_INVALID = 0,
    BLE_MSG_CALIBRATE,
    BLE_MSG_SIMPLE_ROUTE,
    BLE_MSG_TURN_TARGET
} ble_msg_type_t;

typedef struct {
    ble_msg_type_t type;
    servo_cmd_t simple_cmd;
    int turn_sign;   /* LEFT=-1, RIGHT=+1 */
    int turn_deg;    /* any positive degree value, e.g. 17, 63, 92, 180 */
} ble_msg_t;

static QueueHandle_t servo_queue;
static QueueHandle_t ble_queue;

static uint8_t own_addr_type;
static char last_command[32] = "IDLE";

static volatile nav_state_t g_nav_state = NAV_WAITING_CALIBRATE;
static volatile int g_cal_servo_idx = -1;
static volatile float g_target_yaw_deg = 0.0f;
static volatile float g_measured_yaw_deg = 0.0f;

static float g_gyro_z_bias_dps = 0.0f;

static i2c_master_bus_handle_t g_i2c_bus = NULL;
static i2c_master_dev_handle_t g_mpu_dev = NULL;

static TickType_t g_last_debug_tick = 0;

static esp_err_t mpu_read_regs(uint8_t reg, uint8_t *data, size_t len);
static void queue_route_instruction(servo_cmd_t cmd, bool also_forward);

/* ---------------- BLE UUIDs ----------------
   Service:        0xFFF0
   Command char:   0xFFF1
*/
static int gatt_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                          struct ble_gatt_access_ctxt *ctxt, void *arg);

static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = BLE_UUID16_DECLARE(0xFFF0),
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = BLE_UUID16_DECLARE(0xFFF1),
                .access_cb = gatt_access_cb,
                .flags = BLE_GATT_CHR_F_READ |
                         BLE_GATT_CHR_F_WRITE |
                         BLE_GATT_CHR_F_WRITE_NO_RSP,
            },
            {0}
        },
    },
    {0}
};

/* ---------------- Helpers ---------------- */

static void normalize_command(char *s)
{
    size_t len = strlen(s);

    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        s[--len] = '\0';
    }

    size_t start = 0;
    while (s[start] && isspace((unsigned char)s[start])) {
        start++;
    }

    if (start > 0) {
        memmove(s, s + start, strlen(s + start) + 1);
    }

    for (size_t i = 0; s[i]; i++) {
        s[i] = (char)toupper((unsigned char)s[i]);
    }
}

static bool parse_simple_route_command(const char *cmd, servo_cmd_t *out)
{
    if (strcmp(cmd, "LEFT") == 0) {
        *out = CMD_LEFT;
        return true;
    }
    if (strcmp(cmd, "RIGHT") == 0) {
        *out = CMD_RIGHT;
        return true;
    }
    if (strcmp(cmd, "FORWARD") == 0) {
        *out = CMD_FORWARD;
        return true;
    }
    if (strcmp(cmd, "BACKWARD") == 0) {
        *out = CMD_BACKWARD;
        return true;
    }
    return false;
}

static bool parse_ble_message(const char *cmd, ble_msg_t *out)
{
    memset(out, 0, sizeof(*out));

    if (strcmp(cmd, "CALIBRATE") == 0) {
        out->type = BLE_MSG_CALIBRATE;
        return true;
    }

    /* turn target commands: LEFT,17 / RIGHT,92 / LEFT,180 */
    char dir[16] = {0};
    int deg = 0;

    if (sscanf(cmd, "%15[^,],%d", dir, &deg) == 2) {
        normalize_command(dir);

        /* allow any positive degree value */
        if (deg > 0 && strcmp(dir, "LEFT") == 0) {
            out->type = BLE_MSG_TURN_TARGET;
            out->turn_sign = -1;
            out->turn_deg = deg;
            return true;
        }

        if (deg > 0 && strcmp(dir, "RIGHT") == 0) {
            out->type = BLE_MSG_TURN_TARGET;
            out->turn_sign = +1;
            out->turn_deg = deg;
            return true;
        }
    }

    servo_cmd_t simple;
    if (parse_simple_route_command(cmd, &simple)) {
        out->type = BLE_MSG_SIMPLE_ROUTE;
        out->simple_cmd = simple;
        return true;
    }

    return false;
}

static void status_set(const char *s)
{
    snprintf(last_command, sizeof(last_command), "%s", s);
}

static uint32_t servo_angle_to_duty(int angle_deg)
{
    if (angle_deg < 0) angle_deg = 0;
    if (angle_deg > 180) angle_deg = 180;

    uint32_t pulse_us = SERVO_MIN_PULSE_US +
        ((uint32_t)angle_deg * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US)) / 180;

    uint32_t max_duty = (1U << SERVO_LEDC_RES) - 1U;
    uint32_t duty = (pulse_us * max_duty) / SERVO_PERIOD_US;

    return duty;
}

static void servo_write_angle(int servo_idx, int angle_deg)
{
    uint32_t duty = servo_angle_to_duty(angle_deg);
    ledc_set_duty(SERVO_LEDC_MODE, servo_channel[servo_idx], duty);
    ledc_update_duty(SERVO_LEDC_MODE, servo_channel[servo_idx]);
}

static void servo_all_idle(void)
{
    for (int i = 0; i < SERVO_COUNT; i++) {
        servo_write_angle(i, SERVO_IDLE_DEG);
    }
}

static void servo_pwm_init(void)
{
    ledc_timer_config_t timer_cfg = {
        .speed_mode = SERVO_LEDC_MODE,
        .duty_resolution = SERVO_LEDC_RES,
        .timer_num = SERVO_LEDC_TIMER,
        .freq_hz = SERVO_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer_cfg));

    for (int i = 0; i < SERVO_COUNT; i++) {
        ledc_channel_config_t ch_cfg = {
            .gpio_num = servo_gpio[i],
            .speed_mode = SERVO_LEDC_MODE,
            .channel = servo_channel[i],
            .timer_sel = SERVO_LEDC_TIMER,
            .duty = 0,
            .hpoint = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch_cfg));
    }

    servo_all_idle();
}

static void queue_forward_after_turn(void)
{
#if FORWARD_AFTER_TURN_ENABLE
    servo_cmd_t cmd = CMD_FORWARD;

    for (int i = 0; i < FORWARD_AFTER_TURN_COUNT; i++) {
        if (xQueueSend(servo_queue, &cmd, 0) != pdTRUE) {
            ESP_LOGW(TAG, "servo_queue full while queueing local forward");
            break;
        }
    }
#endif
}

static void queue_route_instruction(servo_cmd_t cmd, bool also_forward)
{
    if (xQueueSend(servo_queue, &cmd, 0) != pdTRUE) {
        ESP_LOGW(TAG, "servo_queue full while queueing route instruction");
        return;
    }

    g_last_route_cmd = cmd;
    g_have_last_cmd  = true;

    if (also_forward) {
        queue_forward_after_turn();
    }
}

static esp_err_t mpu_read_6axis(mpu_sample_t *s)
{
    uint8_t raw[14];
    esp_err_t err = mpu_read_regs(MPU_REG_ACCEL_XOUT_H, raw, 14);
    if (err != ESP_OK) {
        return err;
    }

    s->ax = (int16_t)((raw[0]  << 8) | raw[1]);
    s->ay = (int16_t)((raw[2]  << 8) | raw[3]);
    s->az = (int16_t)((raw[4]  << 8) | raw[5]);
    // raw[6], raw[7] = temperature, ignore for now
    s->gx = (int16_t)((raw[8]  << 8) | raw[9]);
    s->gy = (int16_t)((raw[10] << 8) | raw[11]);
    s->gz = (int16_t)((raw[12] << 8) | raw[13]);

    return ESP_OK;
}

/* ---------------- MPU-6050 helpers ---------------- */

static esp_err_t mpu_write_reg(uint8_t reg, uint8_t val)
{
    uint8_t buf[2] = {reg, val};
    return i2c_master_transmit(g_mpu_dev, buf, sizeof(buf), 100);
}

static esp_err_t mpu_read_regs(uint8_t reg, uint8_t *data, size_t len)
{
    return i2c_master_transmit_receive(g_mpu_dev, &reg, 1, data, len, 100);
}

static esp_err_t mpu_read_gyro_z_dps(float *gz_dps)
{
    uint8_t raw[2];
    esp_err_t err = mpu_read_regs(MPU_REG_GYRO_ZOUT_H, raw, 2);
    if (err != ESP_OK) {
        return err;
    }

    int16_t raw_z = (int16_t)((raw[0] << 8) | raw[1]);
    *gz_dps = ((float)raw_z) / MPU_GYRO_SENS_LSB_PER_DPS;
    return ESP_OK;
}

static esp_err_t mpu_calibrate_gyro_bias(float *bias_dps)
{
    float sum = 0.0f;
    int good = 0;

    for (int i = 0; i < MPU_BIAS_SAMPLES; i++) {
        float gz = 0.0f;
        if (mpu_read_gyro_z_dps(&gz) == ESP_OK) {
            sum += gz;
            good++;
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }

    if (good == 0) {
        return ESP_FAIL;
    }

    *bias_dps = sum / (float)good;
    return ESP_OK;
}

static esp_err_t mpu_init(void)
{
    i2c_master_bus_config_t bus_cfg = {
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .i2c_port = MPU_I2C_PORT,
        .scl_io_num = MPU_SCL_GPIO,
        .sda_io_num = MPU_SDA_GPIO,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &g_i2c_bus));
    ESP_ERROR_CHECK(i2c_master_probe(g_i2c_bus, MPU_ADDR, 100));

    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MPU_ADDR,
        .scl_speed_hz = 400000,
    };
    ESP_ERROR_CHECK(i2c_master_bus_add_device(g_i2c_bus, &dev_cfg, &g_mpu_dev));

    uint8_t who = 0;
    ESP_ERROR_CHECK(mpu_read_regs(MPU_REG_WHO_AM_I, &who, 1));
    ESP_LOGI(TAG, "MPU WHO_AM_I = 0x%02X", who);

    /* wake device, set clock source */
    ESP_ERROR_CHECK(mpu_write_reg(MPU_REG_PWR_MGMT_1, 0x01));
    vTaskDelay(pdMS_TO_TICKS(100));

    /* gyro full-scale = +-500 dps */
    ESP_ERROR_CHECK(mpu_write_reg(MPU_REG_GYRO_CONFIG, 0x08));
    vTaskDelay(pdMS_TO_TICKS(20));

    /* accel full-scale = +-2g */
    ESP_ERROR_CHECK(mpu_write_reg(MPU_REG_ACCEL_CONFIG, 0x00));
    vTaskDelay(pdMS_TO_TICKS(20));
    return ESP_OK;
}

/* ---------------- Tasks ---------------- */

static void servo_task(void *arg)
{
    (void)arg;
    servo_cmd_t cmd;
    bool buzz_flip = false;

    while (1) {
        /* continuous buzz during calibration */
        if (g_nav_state == NAV_CALIBRATING && g_cal_servo_idx >= 0) {
            int idx = g_cal_servo_idx;
            servo_write_angle(idx, buzz_flip ? SERVO_BUZZ_HIGH_DEG : SERVO_BUZZ_LOW_DEG);
            buzz_flip = !buzz_flip;
            vTaskDelay(pdMS_TO_TICKS(SERVO_BUZZ_MS));
            continue;
        }

        /* otherwise do normal pulsed route actuation */
        if (xQueueReceive(servo_queue, &cmd, pdMS_TO_TICKS(40)) == pdTRUE) {
            int idx = (int)cmd;

            ESP_LOGI(TAG, "Actuating route servo %d", idx);

            for (int i = 0; i < SERVO_PULSES; i++) {
                servo_write_angle(idx, SERVO_ACTIVE_DEG);
                vTaskDelay(pdMS_TO_TICKS(SERVO_PULSE_MS));
                servo_write_angle(idx, SERVO_IDLE_DEG);
                vTaskDelay(pdMS_TO_TICKS(120));
            }
        }
    }
}

static void control_task(void *arg)
{
    (void)arg;
    ble_msg_t msg;

    while (1) {
        if (xQueueReceive(ble_queue, &msg, portMAX_DELAY) != pdTRUE) {
            continue;
        }

        switch (msg.type) {
        case BLE_MSG_CALIBRATE:
            ESP_LOGI(TAG, "CALIBRATE received. Hold still...");
            status_set("CAL_HOLD_STILL");

            g_nav_state = NAV_WAITING_TURN_TARGET;
            g_cal_servo_idx = -1;
            g_target_yaw_deg = 0.0f;
            g_measured_yaw_deg = 0.0f;
            servo_all_idle();

            if (mpu_calibrate_gyro_bias(&g_gyro_z_bias_dps) == ESP_OK) {
                ESP_LOGI(TAG, "Gyro Z bias = %.3f dps", g_gyro_z_bias_dps);
                status_set("CAL_READY");
            } else {
                ESP_LOGE(TAG, "Gyro bias calibration failed");
                status_set("CAL_FAIL");
                g_nav_state = NAV_WAITING_CALIBRATE;
            }
            break;

            case BLE_MSG_TURN_TARGET: {
                /* After startup calibration is done, ignore degree and use only direction */
                if (g_nav_state == NAV_ROUTING) {
                    servo_cmd_t route_cmd = (msg.turn_sign < 0) ? CMD_LEFT : CMD_RIGHT;
                    queue_route_instruction(route_cmd, true);
                    status_set("ROUTE_TURN");
                    break;
                }
            
                if (g_nav_state != NAV_WAITING_TURN_TARGET) {
                    ESP_LOGW(TAG, "Ignoring turn target because CALIBRATE has not armed calibration");
                    status_set("NEED_CALIBRATE");
                    break;
                }
            
                int bounded_deg = msg.turn_deg;
            
                if (bounded_deg < 1) {
                    bounded_deg = 1;
                }
                if (bounded_deg > 360) {
                    bounded_deg = 360;
                }
            
                g_target_yaw_deg = (float)(msg.turn_sign * bounded_deg);
                g_measured_yaw_deg = 0.0f;
                g_cal_servo_idx = (msg.turn_sign < 0) ? CMD_LEFT : CMD_RIGHT;
                g_nav_state = NAV_CALIBRATING;
            
                ESP_LOGI(TAG, "Calibration target = %.1f deg, buzzing servo %d",
                         g_target_yaw_deg, g_cal_servo_idx);
                status_set("CAL_TURNING");
                break;
            }

        case BLE_MSG_SIMPLE_ROUTE:
            if (g_nav_state != NAV_ROUTING) {
                ESP_LOGW(TAG, "Ignoring route cmd until calibration is complete");
                status_set("WAIT_CAL_DONE");
                break;
            }

            if (xQueueSend(servo_queue, &msg.simple_cmd, 0) != pdTRUE) {
                ESP_LOGW(TAG, "servo_queue full");
            }
            break;

        default:
            break;
        }
    }
}

static void mpu_task(void *arg)
{
    (void)arg;
    TickType_t last = xTaskGetTickCount();

    while (1) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(MPU_SAMPLE_MS));

        mpu_sample_t s;
        if (mpu_read_6axis(&s) != ESP_OK) {
            ESP_LOGW(TAG, "MPU read failed");
            continue;
        }

        taskENTER_CRITICAL(&g_ring_mux);
        g_ring[g_ring_head][0] = s.ax;
        g_ring[g_ring_head][1] = s.ay;
        g_ring[g_ring_head][2] = s.az;
        g_ring[g_ring_head][3] = s.gx;
        g_ring[g_ring_head][4] = s.gy;
        g_ring[g_ring_head][5] = s.gz;
        g_ring_head = (g_ring_head + 1) % GESTURE_WINDOW;
        if (g_ring_head == 0) g_ring_full = true;
        taskEXIT_CRITICAL(&g_ring_mux);

        int64_t now_ms = esp_timer_get_time() / 1000;

        if (g_log_enabled && (now_ms - g_log_start_ms) <= LOG_WINDOW_MS) {
            printf("CSV,%lld,%d,%d,%d,%d,%d,%d,%d\n",
                   now_ms,
                   g_log_label,
                   s.ax, s.ay, s.az,
                   s.gx, s.gy, s.gz);
        } else if (g_log_enabled) {
            g_log_enabled = false;
            printf("LOG_DONE\n");
            fflush(stdout);
        }

        if (g_nav_state == NAV_CALIBRATING) {
            float gz_dps = ((float)s.gz) / MPU_GYRO_SENS_LSB_PER_DPS;
            float yaw_rate_dps = MPU_YAW_SIGN * (gz_dps - g_gyro_z_bias_dps);
            float dt_s = ((float)MPU_SAMPLE_MS) / 1000.0f;
            g_measured_yaw_deg += yaw_rate_dps * dt_s;

            TickType_t now = xTaskGetTickCount();
            if ((now - g_last_debug_tick) >= pdMS_TO_TICKS(100)) {
                g_last_debug_tick = now;
                ESP_LOGI(TAG,
                         "GYRO DEBUG: raw_z=%.2f dps, bias=%.2f dps, corrected=%.2f dps, measured=%.2f deg, target=%.2f deg",
                         gz_dps,
                         g_gyro_z_bias_dps,
                         yaw_rate_dps,
                         g_measured_yaw_deg,
                         g_target_yaw_deg);
            }

            if (fabsf(g_target_yaw_deg - g_measured_yaw_deg) <= CAL_TOL_DEG) {
                ESP_LOGI(TAG, "Calibration turn reached: target=%.1f measured=%.1f",
                         g_target_yaw_deg, g_measured_yaw_deg);

                g_nav_state = NAV_ROUTING;
                g_cal_servo_idx = -1;
                servo_all_idle();
                status_set("CAL_DONE");

                vTaskDelay(pdMS_TO_TICKS(FORWARD_AFTER_TURN_DELAY_MS));
                queue_forward_after_turn();
                status_set("LOCAL_FORWARD");
            }
        }
    }
}

/* ---------------- BLE GATT access ---------------- */

static int gatt_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                          struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;

    if (ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR) {
        int rc = os_mbuf_append(ctxt->om, last_command, strlen(last_command));
        return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
    }

    if (ctxt->op == BLE_GATT_ACCESS_OP_WRITE_CHR) {
        char buf[32];
        uint16_t out_len = 0;

        int rc = ble_hs_mbuf_to_flat(ctxt->om, buf, sizeof(buf) - 1, &out_len);
        if (rc != 0) {
            return BLE_ATT_ERR_UNLIKELY;
        }

        buf[out_len] = '\0';
        normalize_command(buf);

        ESP_LOGI(TAG, "BLE RX: '%s'", buf);
        status_set(buf);

        ble_msg_t msg;
        if (!parse_ble_message(buf, &msg)) {
            ESP_LOGW(TAG, "Unknown BLE command: %s", buf);
            status_set("BAD_CMD");
            return 0;
        }

        if (xQueueSend(ble_queue, &msg, 0) != pdTRUE) {
            ESP_LOGW(TAG, "ble_queue full");
            status_set("QUEUE_FULL");
        }

        return 0;
    }

    return BLE_ATT_ERR_UNLIKELY;
}

/* ---------------- BLE advertising / GAP ---------------- */

static void ble_advertise(void);

static int gap_event_cb(struct ble_gap_event *event, void *arg)
{
    (void)arg;

    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        ESP_LOGI(TAG, "BLE connect %s, status=%d",
                 event->connect.status == 0 ? "ok" : "failed",
                 event->connect.status);
        if (event->connect.status != 0) {
            ble_advertise();
        }
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGI(TAG, "BLE disconnected, reason=%d", event->disconnect.reason);
        ble_advertise();
        return 0;

    case BLE_GAP_EVENT_ADV_COMPLETE:
        ble_advertise();
        return 0;

    default:
        return 0;
    }
}

static void ble_on_reset(int reason)
{
    ESP_LOGE(TAG, "BLE reset, reason=%d", reason);
}

static void ble_on_sync(void)
{
    int rc = ble_hs_util_ensure_addr(0);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_util_ensure_addr failed: %d", rc);
        return;
    }

    rc = ble_hs_id_infer_auto(0, &own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed: %d", rc);
        return;
    }

    ble_advertise();
}

static void ble_advertise(void)
{
    struct ble_gap_adv_params adv_params;
    struct ble_hs_adv_fields fields;
    int rc;

    memset(&fields, 0, sizeof(fields));

    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.tx_pwr_lvl_is_present = 1;
    fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;

    const char *name = ble_svc_gap_device_name();
    fields.name = (const uint8_t *)name;
    fields.name_len = strlen(name);
    fields.name_is_complete = 1;

    ble_uuid16_t svc_uuid = BLE_UUID16_INIT(0xFFF0);
    fields.uuids16 = &svc_uuid;
    fields.num_uuids16 = 1;
    fields.uuids16_is_complete = 1;

    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
        return;
    }

    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

    rc = ble_gap_adv_start(own_addr_type, NULL, BLE_HS_FOREVER,
                           &adv_params, gap_event_cb, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
    } else {
        ESP_LOGI(TAG, "Advertising as '%s'", name);
    }
}

static void ble_host_task(void *param)
{
    (void)param;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

/* ---------------- app_main ---------------- */
static void gesture_task(void *arg)
{
    (void)arg;
    static float window[GESTURE_WINDOW * GESTURE_AXES];
    TickType_t last = xTaskGetTickCount();

    while (1) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(GESTURE_STRIDE_MS));

        if (!g_ring_full) continue;

        // ---- 1. Snapshot ring buffer (oldest-first) ----
        taskENTER_CRITICAL(&g_ring_mux);
        int head = g_ring_head;
        int16_t snapshot[GESTURE_WINDOW][GESTURE_AXES];
        for (int i = 0; i < GESTURE_WINDOW; i++) {
            int src = (head + i) % GESTURE_WINDOW;
            for (int ax = 0; ax < GESTURE_AXES; ax++) {
                snapshot[i][ax] = g_ring[src][ax];
            }
        }
        taskEXIT_CRITICAL(&g_ring_mux);

        // ---- 2. Motion gate: skip classification if there's no real motion ----
        float motion_sum = 0.0f;
        for (int i = 1; i < GESTURE_WINDOW; i++) {
            for (int ax = 0; ax < GESTURE_AXES; ax++) {
                float d = (float)snapshot[i][ax] - (float)snapshot[i-1][ax];
                motion_sum += d * d;
            }
        }

        // Log motion periodically so we can tune MOTION_FLOOR
        // static int dbg_count = 0;
        // if ((dbg_count++ % 4) == 0) {
        //     ESP_LOGI(TAG, "motion_sum=%.0f", motion_sum);
        // }

        if (motion_sum < MOTION_FLOOR) continue;

        // ---- 3. Normalize and classify ----
        for (int i = 0; i < GESTURE_WINDOW; i++) {
            for (int ax = 0; ax < GESTURE_AXES; ax++) {
                window[i * GESTURE_AXES + ax] =
                    ((float)snapshot[i][ax] - NORM_MEAN[ax]) / NORM_STD[ax];
            }
        }

        int idle_logit = 0, tap_logit = 0;
        if (gesture_classify_logits(window, &idle_logit, &tap_logit) != 0) continue;

        int margin = tap_logit - idle_logit;
        int64_t now_ms = esp_timer_get_time() / 1000;
        bool confident_tap  = margin >= TAP_MARGIN;
        bool cooldown_clear = (now_ms - last_tap_fire_ms) > TAP_COOLDOWN_MS;

        // ---- 4. Fire ----
        if (confident_tap && cooldown_clear) {
            last_tap_fire_ms = now_ms;
            ESP_LOGI(TAG, "TAP (margin=%d, motion=%.0f)", margin, motion_sum);

            if (g_nav_state == NAV_ROUTING && g_have_last_cmd) {
                servo_cmd_t replay = (servo_cmd_t)g_last_route_cmd;
                if (xQueueSend(servo_queue, &replay, 0) == pdTRUE) {
                    status_set("TAP_REPLAY");
                }
            }
        }
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    servo_queue = xQueueCreate(8, sizeof(servo_cmd_t));
    ble_queue   = xQueueCreate(8, sizeof(ble_msg_t));
    if (!servo_queue || !ble_queue) {
        ESP_LOGE(TAG, "Failed to create queues");
        return;
    }

    servo_pwm_init();
    ESP_ERROR_CHECK(mpu_init());

    xTaskCreate(servo_task,   "servo_task",   4096, NULL, 5, NULL);
    xTaskCreate(control_task, "control_task", 4096, NULL, 5, NULL);
    xTaskCreate(mpu_task,     "mpu_task",     4096, NULL, 5, NULL);
    xTaskCreate(gesture_task, "gesture_task", 8192, NULL, 4, NULL);

    ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: %d", ret);
        return;
    }

    ble_hs_cfg.reset_cb = ble_on_reset;
    ble_hs_cfg.sync_cb  = ble_on_sync;

    ble_svc_gap_init();
    ble_svc_gatt_init();
    ble_store_config_init();

    int rc = ble_svc_gap_device_name_set("ESP32S3-Servo");
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_svc_gap_device_name_set failed: %d", rc);
        return;
    }

    rc = ble_gatts_count_cfg(gatt_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
        return;
    }

    rc = ble_gatts_add_svcs(gatt_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
        return;
    }

    status_set("WAIT_CAL");
    nimble_port_freertos_init(ble_host_task);

    g_log_enabled = false;
    g_log_start_ms = esp_timer_get_time() / 1000;
    printf("BOOT_START\n");
    fflush(stdout);

    ESP_ERROR_CHECK(gesture_model_init());
}