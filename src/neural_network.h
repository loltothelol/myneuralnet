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

struct neural_layer {
	const char *name;
	unsigned size;
	neural_link *prev, *next;
};

struct neural_link {
	neural_layer *source, *target;
	double weights[];
};

/////////////////////////////////
//////// LAYER FUNCTIONS ////////
/////////////////////////////////

extern neural_layer *create_neural_layer(unsigned size, const char *name);
extern neural_link *link_neural_layers(neural_layer *source, neural_layer *target);

////////////////////////////////
//////// LINK FUNCTIONS ////////
////////////////////////////////

static unsigned count_link_weights(const neural_link *link) {
	return link->source->size * link->target->size;
}

extern void randomize_weights(neural_link *link);
extern void feed_forward(const neural_link *across, double *input, double *output);

/////////////////////////////////
//////// DEBUG FUNCTIONS ////////
/////////////////////////////////

extern void print_neural_layer(const neural_layer *layer);
extern void print_neural_link(const neural_link *link);
