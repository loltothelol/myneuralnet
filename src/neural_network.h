#pragma once

#include <stdlib.h>

static double rand_double(double min, double max) {
	double r = (double)rand() / (double)RAND_MAX;
	return r * (max - min) + min;
}

////////////////////////////
//////// STRUCTURES ////////
////////////////////////////

typedef struct neural_layer neural_layer;
typedef struct neural_link neural_link;
typedef struct neural_network neural_network;

struct neural_layer {
	const char *name;
	unsigned size;
	neural_link *prev, *next;
};

struct neural_link {
	neural_layer *source, *target;
	double weights[];
};

struct neural_network {
	neural_layer *start, *end;
};

/////////////////////////////////
//////// LAYER FUNCTIONS ////////
/////////////////////////////////

extern neural_layer *create_neural_layer(unsigned size, const char *name);
extern neural_link *link_neural_layers(neural_layer *source, neural_layer *target);

////////////////////////////////
//////// LINK FUNCTIONS ////////
////////////////////////////////

static inline unsigned count_weights(neural_layer *source, neural_layer *target) {
	// (source neurons + bias neuron [constant = 1.0]) * target neurons
	return (source->size + 1) * target->size;
}

static inline unsigned count_link_weights(const neural_link *link) {
	return count_weights(link->source, link->target);
}

extern void randomize_weights(neural_link *link);
extern void feed_forward(const neural_link *across, double *input, double *output);
extern void feed_backward(const neural_link *across, double *input, double *output);

/////////////////////////////////
//////// DEBUG FUNCTIONS ////////
/////////////////////////////////

extern void print_neural_layer(const neural_layer *layer);
extern void print_neural_link(const neural_link *link);
