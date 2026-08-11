#include "image_operation_plugin.h"

#include <stdlib.h>
#include <string.h>

struct exports_image_rs_plugin_image_operation_operation_invocation_t {
  uint32_t processed_planes;
  uint32_t total_planes;
  uint8_t mode;
  image_rs_plugin_types_image_metadata_t finish_metadata;
};

enum fixture_mode {
  MODE_ADD_ONE,
  MODE_FAIL_FINISH,
  MODE_SPIN,
  MODE_GROW_MEMORY,
  MODE_BAD_PROGRESS,
  MODE_BAD_REPLACEMENT,
};

static uint8_t supported_pixel_types[] = {
    IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_UINT8,
    IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_UINT16,
    IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_FLOAT32,
};

static uint8_t supported_scopes[] = {
    EXPORTS_IMAGE_RS_PLUGIN_IMAGE_OPERATION_PLANE_SCOPE_ALL_PLANES,
};

static bool string_equals(const image_operation_plugin_string_t *value,
                          const char *expected) {
  size_t expected_length = strlen(expected);
  if (value->len != expected_length) {
    return false;
  }
  for (size_t index = 0; index < expected_length; index++) {
    if (value->ptr[index] != (uint8_t)expected[index]) {
      return false;
    }
  }
  return true;
}

static bool is_supported_entrypoint(
    const image_operation_plugin_string_t *entrypoint) {
  return string_equals(entrypoint, "add-one") ||
         string_equals(entrypoint, "fail-finish") ||
         string_equals(entrypoint, "spin") ||
         string_equals(entrypoint, "grow-memory") ||
         string_equals(entrypoint, "bad-progress") ||
         string_equals(entrypoint, "bad-replacement") ||
         string_equals(entrypoint, "needs-roi");
}

static uint8_t mode_for_entrypoint(
    const image_operation_plugin_string_t *entrypoint) {
  if (string_equals(entrypoint, "fail-finish")) {
    return MODE_FAIL_FINISH;
  }
  if (string_equals(entrypoint, "spin")) {
    return MODE_SPIN;
  }
  if (string_equals(entrypoint, "grow-memory")) {
    return MODE_GROW_MEMORY;
  }
  if (string_equals(entrypoint, "bad-progress")) {
    return MODE_BAD_PROGRESS;
  }
  if (string_equals(entrypoint, "bad-replacement")) {
    return MODE_BAD_REPLACEMENT;
  }
  return MODE_ADD_ONE;
}

static void set_error(
    exports_image_rs_plugin_image_operation_plugin_error_t *error,
    image_rs_plugin_types_error_kind_t kind, const char *message,
    const char *details_json) {
  error->kind = kind;
  image_operation_plugin_string_set(&error->message, message);
  error->details_json.is_some = details_json != NULL;
  if (details_json != NULL) {
    image_operation_plugin_string_set(&error->details_json.val, details_json);
  }
}

static bool copy_string(const image_operation_plugin_string_t *source,
                        image_operation_plugin_string_t *destination) {
  destination->ptr = NULL;
  destination->len = 0;
  if (source->len == 0) {
    return true;
  }
  destination->ptr = malloc(source->len);
  if (destination->ptr == NULL) {
    return false;
  }
  memcpy(destination->ptr, source->ptr, source->len);
  destination->len = source->len;
  return true;
}

static bool copy_finish_metadata(
    const image_rs_plugin_types_image_metadata_t *source,
    image_rs_plugin_types_image_metadata_t *destination) {
  memset(destination, 0, sizeof(*destination));

  if (source->dimensions.len >
      SIZE_MAX / sizeof(*destination->dimensions.ptr)) {
    return false;
  }
  if (source->dimensions.len != 0) {
    destination->dimensions.ptr =
        malloc(source->dimensions.len * sizeof(*destination->dimensions.ptr));
    if (destination->dimensions.ptr == NULL) {
      return false;
    }
    memset(destination->dimensions.ptr, 0,
           source->dimensions.len * sizeof(*destination->dimensions.ptr));
    destination->dimensions.len = source->dimensions.len;
    for (size_t index = 0; index < source->dimensions.len; index++) {
      destination->dimensions.ptr[index] = source->dimensions.ptr[index];
      destination->dimensions.ptr[index].unit.val.ptr = NULL;
      destination->dimensions.ptr[index].unit.val.len = 0;
      if (source->dimensions.ptr[index].unit.is_some &&
          !copy_string(&source->dimensions.ptr[index].unit.val,
                       &destination->dimensions.ptr[index].unit.val)) {
        image_rs_plugin_types_image_metadata_free(destination);
        memset(destination, 0, sizeof(*destination));
        return false;
      }
      if (destination->dimensions.ptr[index].axis ==
          IMAGE_RS_PLUGIN_TYPES_AXIS_KIND_X) {
        destination->dimensions.ptr[index].spacing.is_some = true;
        destination->dimensions.ptr[index].spacing.val = 1.25;
      }
    }
  }

  if (source->channel_names.len >
      SIZE_MAX / sizeof(*destination->channel_names.ptr)) {
    image_rs_plugin_types_image_metadata_free(destination);
    memset(destination, 0, sizeof(*destination));
    return false;
  }
  if (source->channel_names.len != 0) {
    destination->channel_names.ptr =
        malloc(source->channel_names.len * sizeof(*destination->channel_names.ptr));
    if (destination->channel_names.ptr == NULL) {
      image_rs_plugin_types_image_metadata_free(destination);
      memset(destination, 0, sizeof(*destination));
      return false;
    }
    memset(destination->channel_names.ptr, 0,
           source->channel_names.len * sizeof(*destination->channel_names.ptr));
    destination->channel_names.len = source->channel_names.len;
    for (size_t index = 0; index < source->channel_names.len; index++) {
      if (!copy_string(&source->channel_names.ptr[index],
                       &destination->channel_names.ptr[index])) {
        image_rs_plugin_types_image_metadata_free(destination);
        memset(destination, 0, sizeof(*destination));
        return false;
      }
    }
  }

  if (source->properties.len >
      SIZE_MAX / sizeof(*destination->properties.ptr)) {
    image_rs_plugin_types_image_metadata_free(destination);
    memset(destination, 0, sizeof(*destination));
    return false;
  }
  if (source->properties.len != 0) {
    destination->properties.ptr =
        malloc(source->properties.len * sizeof(*destination->properties.ptr));
    if (destination->properties.ptr == NULL) {
      image_rs_plugin_types_image_metadata_free(destination);
      memset(destination, 0, sizeof(*destination));
      return false;
    }
    memset(destination->properties.ptr, 0,
           source->properties.len * sizeof(*destination->properties.ptr));
    destination->properties.len = source->properties.len;
    for (size_t index = 0; index < source->properties.len; index++) {
      if (!copy_string(&source->properties.ptr[index].name,
                       &destination->properties.ptr[index].name) ||
          !copy_string(&source->properties.ptr[index].value_json,
                       &destination->properties.ptr[index].value_json)) {
        image_rs_plugin_types_image_metadata_free(destination);
        memset(destination, 0, sizeof(*destination));
        return false;
      }
    }
  }
  return true;
}

static bool copy_and_increment(
    const exports_image_rs_plugin_image_operation_plane_buffer_t *input,
    exports_image_rs_plugin_image_operation_plane_buffer_t *output,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  size_t bytes_per_sample;
  switch (input->sample_type) {
  case IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_UINT8:
    bytes_per_sample = 1;
    break;
  case IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_UINT16:
    bytes_per_sample = 2;
    break;
  case IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_FLOAT32:
    bytes_per_sample = 4;
    break;
  default:
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_UNSUPPORTED_IMAGE,
              "unsupported pixel type", NULL);
    return false;
  }

  uint64_t expected_length =
      (uint64_t)input->width * input->height * bytes_per_sample;
  if (expected_length != input->pixels.len) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_BUFFER,
              "plane byte length does not match its dimensions", NULL);
    return false;
  }

  uint8_t *pixels = malloc(input->pixels.len);
  if (pixels == NULL) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_RESOURCE_LIMIT,
              "fixture allocation failed", NULL);
    return false;
  }
  memcpy(pixels, input->pixels.ptr, input->pixels.len);

  if (input->sample_type == IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_UINT8) {
    for (size_t offset = 0; offset < input->pixels.len; offset++) {
      pixels[offset] = (uint8_t)(pixels[offset] + 1);
    }
  } else if (input->sample_type == IMAGE_RS_PLUGIN_TYPES_PIXEL_TYPE_UINT16) {
    for (size_t offset = 0; offset < input->pixels.len; offset += 2) {
      uint16_t value =
          (uint16_t)pixels[offset] | ((uint16_t)pixels[offset + 1] << 8);
      value = (uint16_t)(value + 1);
      pixels[offset] = (uint8_t)value;
      pixels[offset + 1] = (uint8_t)(value >> 8);
    }
  } else {
    for (size_t offset = 0; offset < input->pixels.len; offset += 4) {
      uint32_t bits = (uint32_t)pixels[offset] |
                      ((uint32_t)pixels[offset + 1] << 8) |
                      ((uint32_t)pixels[offset + 2] << 16) |
                      ((uint32_t)pixels[offset + 3] << 24);
      float value;
      memcpy(&value, &bits, sizeof(value));
      value += 1.0f;
      memcpy(&bits, &value, sizeof(bits));
      pixels[offset] = (uint8_t)bits;
      pixels[offset + 1] = (uint8_t)(bits >> 8);
      pixels[offset + 2] = (uint8_t)(bits >> 16);
      pixels[offset + 3] = (uint8_t)(bits >> 24);
    }
  }

  *output = *input;
  output->pixels.ptr = pixels;
  return true;
}

static bool set_plane_measurement(
    uint32_t plane_number,
    exports_image_rs_plugin_image_operation_list_measurement_row_t *rows,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  exports_image_rs_plugin_image_operation_measurement_row_t *row =
      malloc(sizeof(*row));
  image_rs_plugin_types_measurement_t *measurement =
      malloc(sizeof(*measurement));
  if (row == NULL || measurement == NULL) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_RESOURCE_LIMIT,
              "fixture allocation failed", NULL);
    return false;
  }

  row->label.is_some = false;
  row->values.ptr = measurement;
  row->values.len = 1;
  image_operation_plugin_string_set(&measurement->column, "plane");
  measurement->value.tag = IMAGE_RS_PLUGIN_TYPES_MEASUREMENT_VALUE_INTEGER;
  measurement->value.val.integer = plane_number;
  rows->ptr = row;
  rows->len = 1;
  return true;
}

static bool set_finish_measurements(
    exports_image_rs_plugin_image_operation_list_measurement_row_t *rows,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  exports_image_rs_plugin_image_operation_measurement_row_t *row =
      malloc(sizeof(*row));
  image_rs_plugin_types_measurement_t *measurements =
      malloc(4 * sizeof(*measurements));
  if (row == NULL || measurements == NULL) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_RESOURCE_LIMIT,
              "fixture allocation failed", NULL);
    return false;
  }

  row->label.is_some = true;
  image_operation_plugin_string_set(&row->label.val, "summary");
  row->values.ptr = measurements;
  row->values.len = 4;

  image_operation_plugin_string_set(&measurements[0].column, "score");
  measurements[0].value.tag = IMAGE_RS_PLUGIN_TYPES_MEASUREMENT_VALUE_NUMBER;
  measurements[0].value.val.number = 1.5;
  image_operation_plugin_string_set(&measurements[1].column, "ok");
  measurements[1].value.tag = IMAGE_RS_PLUGIN_TYPES_MEASUREMENT_VALUE_BOOLEAN;
  measurements[1].value.val.boolean = true;
  image_operation_plugin_string_set(&measurements[2].column, "note");
  measurements[2].value.tag = IMAGE_RS_PLUGIN_TYPES_MEASUREMENT_VALUE_TEXT;
  image_operation_plugin_string_set(&measurements[2].value.val.text, "done");
  image_operation_plugin_string_set(&measurements[3].column, "missing");
  measurements[3].value.tag = IMAGE_RS_PLUGIN_TYPES_MEASUREMENT_VALUE_MISSING;

  rows->ptr = row;
  rows->len = 1;
  return true;
}

bool exports_image_rs_plugin_image_operation_capabilities(
    image_operation_plugin_string_t *entrypoint,
    exports_image_rs_plugin_image_operation_operation_capabilities_t *result,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  if (!is_supported_entrypoint(entrypoint)) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_PARAMETERS,
              "unknown fixture entrypoint", NULL);
    return false;
  }

  result->supported_pixel_types.ptr = supported_pixel_types;
  result->supported_pixel_types.len = 3;
  result->supported_scopes.ptr = supported_scopes;
  result->supported_scopes.len = 1;
  bool needs_roi = string_equals(entrypoint, "needs-roi");
  result->requires_area_roi = needs_roi;
  result->accepts_area_mask = needs_roi;
  result->modifies_pixels = true;
  return true;
}

bool exports_image_rs_plugin_image_operation_begin(
    image_operation_plugin_string_t *entrypoint,
    exports_image_rs_plugin_image_operation_begin_request_t *request,
    exports_image_rs_plugin_image_operation_own_operation_invocation_t *result,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  if (!is_supported_entrypoint(entrypoint)) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_PARAMETERS,
              "unknown fixture entrypoint", NULL);
    return false;
  }
  if (request->selected_scope !=
      EXPORTS_IMAGE_RS_PLUGIN_IMAGE_OPERATION_PLANE_SCOPE_ALL_PLANES) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_PARAMETERS,
              "fixture requires all-planes scope", NULL);
    return false;
  }
  if (request->plane_count == 0) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_PARAMETERS,
              "fixture requires at least one plane", NULL);
    return false;
  }

  exports_image_rs_plugin_image_operation_operation_invocation_t *state =
      malloc(sizeof(*state));
  if (state == NULL) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_RESOURCE_LIMIT,
              "fixture allocation failed", NULL);
    return false;
  }
  state->processed_planes = 0;
  state->total_planes = request->plane_count;
  state->mode = mode_for_entrypoint(entrypoint);
  if (!copy_finish_metadata(&request->image, &state->finish_metadata)) {
    free(state);
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_RESOURCE_LIMIT,
              "fixture metadata allocation failed", NULL);
    return false;
  }
  *result = exports_image_rs_plugin_image_operation_operation_invocation_new(
      state);
  return true;
}

bool exports_image_rs_plugin_image_operation_method_operation_invocation_process_plane(
    exports_image_rs_plugin_image_operation_borrow_operation_invocation_t state,
    exports_image_rs_plugin_image_operation_plane_request_t *request,
    exports_image_rs_plugin_image_operation_plane_output_t *result,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  if (state->mode == MODE_SPIN) {
    volatile uint64_t counter = 0;
    for (;;) {
      counter++;
    }
  }
  if (state->mode == MODE_GROW_MEMORY) {
    volatile uint8_t *allocation = malloc(180U * 1024U * 1024U);
    if (allocation == NULL) {
      set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_RESOURCE_LIMIT,
                "fixture memory allocation failed", NULL);
      return false;
    }
    allocation[0] = 1;
  }
  if (image_rs_plugin_host_is_cancelled()) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_CANCELLED,
              "fixture invocation cancelled", NULL);
    return false;
  }
  if (request->area_roi.is_some) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_PARAMETERS,
              "fixture does not accept an area ROI", NULL);
    return false;
  }
  if (state->processed_planes >= state->total_planes) {
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_PARAMETERS,
              "received more planes than declared", NULL);
    return false;
  }

  result->replacement.is_some = true;
  if (!copy_and_increment(&request->plane, &result->replacement.val, error)) {
    return false;
  }
  if (state->mode == MODE_BAD_REPLACEMENT) {
    result->replacement.val.position.channel++;
  }

  uint32_t completed = state->processed_planes + 1;
  if (!set_plane_measurement(completed, &result->measurements, error)) {
    return false;
  }
  state->processed_planes = completed;

  image_rs_plugin_host_progress_update_t progress;
  progress.completed = completed;
  progress.total.is_some = true;
  progress.total.val = state->total_planes;
  progress.message.is_some = true;
  image_operation_plugin_string_set(&progress.message.val, "add-one");
  image_rs_plugin_host_report_progress(&progress);
  if (state->mode == MODE_BAD_PROGRESS) {
    progress.completed = completed - 1;
    image_rs_plugin_host_report_progress(&progress);
  }
  return true;
}

bool exports_image_rs_plugin_image_operation_finish(
    exports_image_rs_plugin_image_operation_own_operation_invocation_t invocation,
    exports_image_rs_plugin_image_operation_finish_output_t *result,
    exports_image_rs_plugin_image_operation_plugin_error_t *error) {
  exports_image_rs_plugin_image_operation_operation_invocation_t *state =
      exports_image_rs_plugin_image_operation_operation_invocation_rep(
          invocation);
  uint8_t mode = state->mode;
  uint32_t processed_planes = state->processed_planes;
  uint32_t total_planes = state->total_planes;

  if (mode == MODE_FAIL_FINISH) {
    exports_image_rs_plugin_image_operation_operation_invocation_drop_own(
        invocation);
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INTERNAL,
              "intentional finish failure",
              "{\"fixture\":\"fail-finish\"}");
    return false;
  }
  if (processed_planes != total_planes) {
    exports_image_rs_plugin_image_operation_operation_invocation_drop_own(
        invocation);
    set_error(error, IMAGE_RS_PLUGIN_TYPES_ERROR_KIND_INVALID_BUFFER,
              "finish called before every declared plane was processed", NULL);
    return false;
  }

  if (!set_finish_measurements(&result->measurements, error)) {
    exports_image_rs_plugin_image_operation_operation_invocation_drop_own(
        invocation);
    return false;
  }
  result->status.is_some = true;
  image_operation_plugin_string_set(&result->status.val, "add-one complete");
  result->metadata.is_some = true;
  result->metadata.val = state->finish_metadata;
  memset(&state->finish_metadata, 0, sizeof(state->finish_metadata));
  exports_image_rs_plugin_image_operation_operation_invocation_drop_own(
      invocation);
  return true;
}

void exports_image_rs_plugin_image_operation_operation_invocation_destructor(
    exports_image_rs_plugin_image_operation_operation_invocation_t *state) {
  image_rs_plugin_types_image_metadata_free(&state->finish_metadata);
  free(state);
}
