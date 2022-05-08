#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"

static double sigmoid(const double x) {
	return 1.0 / (1.0 + exp(-x));
}

static inline double *get_weight(neural_link *link, unsigned i, unsigned j) {
	return &link->weights[j * link->source->size + i];
}

/////////////////////////////////
//////// LAYER FUNCTIONS ////////
/////////////////////////////////

neural_layer *create_neural_layer(unsigned size, const char *name) {
	neural_layer *const layer = malloc(sizeof(neural_layer));
	*layer = (neural_layer) { .name = name, .size = size };
	return layer;
}

neural_link *link_neural_layers(neural_layer *source, neural_layer *target) {
	// if the source layer is already linked, free it
	if (source->next)
		free(source->next);

	// if the target layer is already linked, free it (if we haven't already above)
	if (target->prev && target->prev != source->next)
		free(target->prev);

	// allocate and initialize the link
	const unsigned weight_count = source->size * target->size;
	neural_link *const link = malloc(sizeof(*link) + weight_count * sizeof(double));
	if (!link) return NULL;
	*link = (neural_link) { .source = source, .target = target };

	// update layer pointers
	source->next = link;
	target->prev = link;

	return link;
}

////////////////////////////////
//////// LINK FUNCTIONS ////////
////////////////////////////////

void randomize_weights(neural_link *link) {
	const unsigned weight_count = count_link_weights(link);
	for (unsigned i = 0; i < weight_count; ++i)
		link->weights[i] = rand_double(-1.0, +1.0);
}

void feed_forward(const neural_link *link, double *input, double *output) {
	neural_layer *const source = link->source;
	neural_layer *const target = link->target;

	// for every target neuron (j)
	for (unsigned j = 0; j < target->size; ++j) {

		// calculate the sum over every source neuron (i)
		double sum = 0.0;
		for (unsigned i = 0; i < source->size; ++i) {
			const double weight = *get_weight((neural_link *) link, i, j);
			const double activation = sigmoid(input[i] * weight);
			sum += activation;
		}

		output[j] = sum;
	}
}

/////////////////////////////////
//////// DEBUG FUNCTIONS ////////
/////////////////////////////////

void print_neural_layer(const neural_layer *layer) {
	if (layer->prev) printf("%s <- ", layer->prev->source->name);
	printf("(%s [%u])", layer->name, layer->size);
	if (layer->next) printf(" -> %s", layer->next->target->name);
	putchar('\n');
}

void print_neural_link(const neural_link *link) {
	if (link->source) printf("\"%s\" <- ", link->source->name);
	printf("([%u])", count_link_weights(link));
	if (link->target) printf(" -> \"%s\"", link->target->name);
	printf(":\n");

	for (unsigned i = 0; i < link->source->size; ++i)
		for (unsigned j = 0; j < link->target->size; ++j)
			printf("\t(i: %u, j: %u) = %f\n", i, j, *get_weight((neural_link *)link, i, j));
}
