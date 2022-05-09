#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural_network.h"

static double sigmoid(const double x) {
	return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_deriv(double x) {
	x = sigmoid(x);
	return x * (1.0 - x);
}

static inline double *access_weight(neural_link_t *link, unsigned i, unsigned j) {
	return &link->weights[j * link->source->size + i];
}

static inline double get_weight(const neural_link_t *link, unsigned i, unsigned j) {
	return *access_weight((neural_link_t *)link, i, j);
}

/////////////////////////////////
//////// LAYER FUNCTIONS ////////
/////////////////////////////////

neural_layer_t *create_neural_layer(unsigned size, const char *name) {
	// allocate and initialize the layer
	const unsigned state_size = size * sizeof(neural_state_t);
	neural_layer_t *layer = malloc(sizeof(neural_layer_t) + state_size);
	*layer = (neural_layer_t) { .name = name, .size = size };
	memset(layer->state, 0, state_size);
	return layer;
}

neural_link_t *link_neural_layers(neural_layer_t *source, neural_layer_t *target) {
	// allocate and initialize the link
	const unsigned weight_count = count_weights(source, target);
	neural_link_t *const link = malloc(sizeof(*link) + weight_count * sizeof(double));
	if (!link) return NULL;
	*link = (neural_link_t) { .source = source, .target = target };

	// if the source layer is already linked, free it
	if (source->next)
		free(source->next);

	// if the target layer is already linked, free it (if we haven't already)
	if (target->prev && target->prev != source->next)
		free(target->prev);

	// update layer pointers
	source->next = link;
	target->prev = link;

	return link;
}

void set_neural_active_state(neural_layer_t *layer, double *active, unsigned size) {
	assert(size == layer->size);
	for (unsigned i = 0; i < size; ++i)
		layer->state[i] = (neural_state_t) { .active = active[i] };
}

void destroy_neural_layer(neural_layer_t *layer) {
	// remove &this from adjacent links
	if (layer->prev) layer->prev->target = NULL;
	if (layer->next) layer->next->source = NULL;
	free(layer);
}

////////////////////////////////
//////// LINK FUNCTIONS ////////
////////////////////////////////

void destroy_neural_link(neural_link_t *link) {
	// remove &this from adjacent layers
	if (link->source) link->source->next = NULL;
	if (link->target) link->target->prev = NULL;
	free(link);
}

void randomize_weights(neural_link_t *link) {
	const unsigned weight_count = count_link_weights(link);
	for (unsigned i = 0; i < weight_count; ++i)
		link->weights[i] = rand_double(-1.0, +1.0);
}

void feed_forward_link(neural_link_t *link) {
	neural_layer_t * source = link->source;
	neural_layer_t * target = link->target;

	// for every target neuron (j)
	for (unsigned j = 0; j < target->size; ++j) {

		// aggregate the bias + sum over every source neuron (i)
		double sum = get_weight(link, source->size, j);
		for (unsigned i = 0; i < source->size; ++i) {
			const double weight_i = get_weight(link, i, j);
			sum += (weight_i * source->state[i].active);
		}

		target->state[j] = (neural_state_t) { sum,  sigmoid(sum) };
	}
}

static double calc_active_deriv(neural_layer_t *layer, unsigned neuron) {
	double sum = 0.0;

	neural_link_t *next_link = layer->next;
	if (!next_link) {
// 		return 2.0 * (calc_active(layer, - y_i));
		if (neuron < layer->size)
			sum = layer->state[neuron].active;

	} else {
		neural_layer_t *next_layer = next_link->target;
		for (unsigned j = 0; j < next_layer->size; ++j) {
			double weight, pre_active, sig_deriv, act_deriv;

			weight = get_weight(next_link, neuron, j);
			pre_active = next_layer->state[j].pre_active;
			sig_deriv = sigmoid_deriv(pre_active);
// 			act_deriv = calc_active_deriv(next_layer, j);
			act_deriv = next_layer->state[j].active_deriv;

			sum += weight * sig_deriv * act_deriv;
		}
	}

	// cache the result iff it isn't a bias neuron
	if (neuron < layer->size)
		layer->state[neuron].active_deriv = sum;

	return sum;
}

static double calc_weight_deriv(neural_layer_t *layer, unsigned j, unsigned k) {
	neural_layer_t *prev_layer = layer->prev->source;
	double active, pre_active, sig_deriv, act_deriv;

	active = (j < prev_layer->size) ? prev_layer->state[j].active : 1.0;
	pre_active = layer->state[k].pre_active;
	sig_deriv = sigmoid_deriv(pre_active);
	act_deriv = calc_active_deriv(layer, j);

	return active * sig_deriv * act_deriv;
}

void feed_backward_layer(neural_layer_t *layer) {
	neural_link_t *link = layer->prev;
	neural_layer_t *prev_layer = link->source;

	for (unsigned j = 0; j < layer->size; ++j) {
		for (unsigned i = 0; i < prev_layer->size + 1; ++i) {
			double deriv = calc_weight_deriv(layer, i, j);

		}
	}
}

/////////////////////////////////
//////// DEBUG FUNCTIONS ////////
/////////////////////////////////

void print_neural_layer(const neural_layer_t *layer) {
	// print basic layer info
	if (layer->prev) printf("%s <- ", layer->prev->source->name);
	printf("(%s [%u])", layer->name, layer->size);
	if (layer->next) printf(" -> %s", layer->next->target->name);
	putchar('\n');

	// print neuron state
	for (unsigned i = 0; i < layer->size; ++i) {
		const neural_state_t *state = &layer->state[i];
		printf("\t- [%u] pre-active: %f, active: %f, active_deriv: %f\n", i, state->pre_active, state->active, state->active_deriv);
	}
	putchar('\n');
}

void print_neural_link(const neural_link_t *link, _Bool header) {
	if (header) {
		if (link->source) printf("\"%s\" <- ", link->source->name);
		printf("([%u])", count_link_weights(link));
		if (link->target) printf(" -> \"%s\"", link->target->name);
		printf(":\n");
	} else printf("weights:\n");

	unsigned i_n = link->source->size;
	unsigned j_n = link->target->size;

	for (unsigned j = 0; j < j_n; ++j) {
		for (unsigned i = 0; i < i_n; ++i)
			printf("\t- (i: %u, j: %u) = %f\n", i, j, get_weight(link, i, j));
		printf("\t- (i: bias, j: %u) = %f\n", j, get_weight(link, i_n, j));
	}


	putchar('\n');
}
