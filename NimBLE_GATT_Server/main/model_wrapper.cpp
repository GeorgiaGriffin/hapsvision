#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "esp_log.h"
#include "esp_check.h"
#include "esp_heap_caps.h"
#include "model_wrapper.h"
#include <cstring>
#include <cstdint>
#include <cmath>

static const char *TAG = "MODEL_WRAPPER";

extern const uint8_t tap_idle_espdl[] asm("_binary_tap_idle_espdl_start");

static dl::Model       *g_model  = nullptr;
static dl::TensorBase  *g_input  = nullptr;
static dl::TensorBase  *g_output = nullptr;

extern "C" esp_err_t gesture_model_init(void)
{
    ESP_LOGI(TAG, "Free internal: %u, largest block: %u",
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    ESP_LOGI(TAG, "Free SPIRAM:   %u, largest block: %u",
        (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
        (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));

    if (g_model == nullptr) {
        g_model = new dl::Model(
            (const char *)tap_idle_espdl,
            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
            0,
            dl::MEMORY_MANAGER_GREEDY,
            nullptr,
            false
        );
    }

    auto inputs  = g_model->get_inputs();
    auto outputs = g_model->get_outputs();
    if (inputs.empty() || outputs.empty()) {
        ESP_LOGE(TAG, "Model has no inputs/outputs");
        return ESP_FAIL;
    }
    g_input  = inputs.begin()->second;
    g_output = outputs.begin()->second;

    ESP_LOGI(TAG, "Input:  name=%s size=%d dtype=%d exp=%d",
             inputs.begin()->first.c_str(),
             g_input->get_size(), (int)g_input->get_dtype(),
             g_input->get_exponent());
    ESP_LOGI(TAG, "Output: name=%s size=%d dtype=%d exp=%d",
             outputs.begin()->first.c_str(),
             g_output->get_size(), (int)g_output->get_dtype(),
             g_output->get_exponent());

    ESP_LOGI(TAG, "Running model self-test...");
    return g_model->test();
}

extern "C" int gesture_classify(const float *window_normalized_600)
{
    if (!g_model || !g_input || !g_output) return -1;

    const int n = g_input->get_size();
    if (n != 600) {
        ESP_LOGE(TAG, "Expected 600 input elements, got %d", n);
        return -1;
    }

    // Quantize floats → int8/int16 using model exponent, or memcpy if float.
    switch (g_input->get_dtype()) {
        case dl::DATA_TYPE_INT8: {
            int8_t *dst = (int8_t *)g_input->get_element_ptr();
            float inv_scale = powf(2.0f, (float)(-g_input->get_exponent()));
            for (int i = 0; i < n; i++) {
                float q = roundf(window_normalized_600[i] * inv_scale);
                if (q > 127.0f)  q = 127.0f;
                if (q < -128.0f) q = -128.0f;
                dst[i] = (int8_t)q;
            }
            break;
        }
        case dl::DATA_TYPE_INT16: {
            int16_t *dst = (int16_t *)g_input->get_element_ptr();
            float inv_scale = powf(2.0f, (float)(-g_input->get_exponent()));
            for (int i = 0; i < n; i++) {
                float q = roundf(window_normalized_600[i] * inv_scale);
                if (q >  32767.0f) q =  32767.0f;
                if (q < -32768.0f) q = -32768.0f;
                dst[i] = (int16_t)q;
            }
            break;
        }
        case dl::DATA_TYPE_FLOAT:
            memcpy(g_input->get_element_ptr(), window_normalized_600,
                   n * sizeof(float));
            break;
        default:
            ESP_LOGE(TAG, "Unsupported input dtype %d", (int)g_input->get_dtype());
            return -1;
    }

    g_model->run();

    // Argmax: both logits share the same exponent, so raw compare == float compare.
    int pred = 0;
    int logit0 = 0, logit1 = 0; 

    switch (g_output->get_dtype()) {
        case dl::DATA_TYPE_INT8: {
            int8_t *o = (int8_t *)g_output->get_element_ptr();
            logit0 = o[0]; logit1 = o[1];       
            if (o[1] > o[0]) pred = 1;
            break;
        }
        case dl::DATA_TYPE_INT16: {
            int16_t *o = (int16_t *)g_output->get_element_ptr();
            logit0 = o[0]; logit1 = o[1];
            if (o[1] > o[0]) pred = 1;
            break;
        }
        case dl::DATA_TYPE_FLOAT: {
            float *o = (float *)g_output->get_element_ptr();
            logit0 = (int)(o[0]*100); logit1 = (int)(o[1]*100);
            if (o[1] > o[0]) pred = 1;
            break;
        }
        default:
            return -1;
    }


    return pred;
}
 
extern "C" int gesture_classify_logits(const float *window_normalized_600,
    int *out_idle, int *out_tap)
{
if (!g_model || !g_input || !g_output) return -1;
if (!out_idle || !out_tap) return -1;

const int n = g_input->get_size();
if (n != 600) return -1;

// Quantize input — same as gesture_classify
if (g_input->get_dtype() == dl::DATA_TYPE_INT8) {
int8_t *dst = (int8_t *)g_input->get_element_ptr();
float inv_scale = powf(2.0f, (float)(-g_input->get_exponent()));
for (int i = 0; i < n; i++) {
float q = roundf(window_normalized_600[i] * inv_scale);
if (q > 127.0f)  q = 127.0f;
if (q < -128.0f) q = -128.0f;
dst[i] = (int8_t)q;
}
} else {
return -1;  // simplify: only int8 path, since that's what your model is
}

g_model->run();

int8_t *o = (int8_t *)g_output->get_element_ptr();
*out_idle = o[0];
*out_tap  = o[1];
return 0;
}