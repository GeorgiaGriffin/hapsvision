#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t gesture_model_init(void);

/* Run inference on a 600-element normalized window (flat, T-major: t0_ax..t0_gz, t1_ax..).
 * Returns 0=idle, 1=tap, or -1 on error. */
int gesture_classify(const float *window_normalized_600);
int gesture_classify_logits(const float *window_normalized_600,
    int *out_idle, int *out_tap);

#ifdef __cplusplus
}
#endif