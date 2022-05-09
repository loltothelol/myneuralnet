#pragma once

#include <stdlib.h>

static double rand_double(double min, double max) {
	double r = (double)rand() / (double)RAND_MAX;
	return r * (max - min) + min;
}

////////////////////////////
//////// STRUCTURES ////////
////////////////////////////

typedef struct neural_state neural_state_t;
typedef struct neural_layer neural_layer_t;
typedef struct neural_link neural_link_t;
typedef struct neural_network neural_network_t;

struct neural_state {
	double pre_active, active, active_deriv;
};

struct neural_layer {
	const char *name;
	unsigned size;
	neural_link_t *prev, *next;
	neural_state_t state[];
};

struct neural_link {
	neural_layer_t *source, *target;
	double weights[];
};

struct neural_network {
	neural_layer_t *start, *end;
};

/////////////////////////////////
//////// LAYER FUNCTIONS ////////
/////////////////////////////////

extern neural_layer_t *create_neural_layer(unsigned size, const char *name);
extern neural_link_t *link_neural_layers(neural_layer_t *source, neural_layer_t *target);
extern void set_neural_active_state(neural_layer_t *layer, double *active, unsigned size);
extern void destroy_neural_layer(neural_layer_t *layer);

////////////////////////////////
//////// LINK FUNCTIONS ////////
////////////////////////////////

extern void destroy_neural_link(neural_link_t *link);

static inline unsigned count_weights(neural_layer_t *source, neural_layer_t *target) {
	// (source neurons + bias neuron [constant = 1.0]) * target neurons
	return (source->size + 1) * target->size;
}

static inline unsigned count_link_weights(const neural_link_t *link) {
	return count_weights(link->source, link->target);
}

extern void randomize_weights(neural_link_t *link);
extern void feed_forward_link(neural_link_t *link);
extern void feed_backward_layer(neural_layer_t *layer);

/////////////////////////////////
//////// DEBUG FUNCTIONS ////////
/////////////////////////////////

extern void print_neural_layer(const neural_layer_t *layer);
extern void print_neural_link(const neural_link_t *link, _Bool header);
