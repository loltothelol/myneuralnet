#include <stdio.h>
#include <time.h>

#include "neural_network.h"

#define INPUT_SIZE 1
#define HIDDEN_SIZE 1
#define OUTPUT_SIZE 1

static void rand_data(double *data, unsigned size) {
	for (unsigned i = 0; i < size; ++i)
		data[i] = rand_double(-1.0, +1.0);
}

static void print_data(double *data, unsigned size) {
	printf("[%u]:\n", size);
	for (unsigned i = 0; i < size; ++i)
		printf("\t(%u) %f\n", i, data[i]);
}

int main() {
	srand(time(NULL));

	// set up a basic network of configuration [ 2, 4, 8 ]

	// create layers

	neural_layer_t *input_layer = create_neural_layer(INPUT_SIZE, "input layer");
	neural_layer_t *hidden_layer = create_neural_layer(HIDDEN_SIZE, "hidden layer");
	neural_layer_t *output_layer = create_neural_layer(OUTPUT_SIZE, "output layer");

	// link layers

	neural_link_t *input_hidden_link = link_neural_layers(input_layer, hidden_layer);
	neural_link_t *hidden_output_link = link_neural_layers(hidden_layer, output_layer);

	// randomize link weights

	randomize_weights(input_hidden_link);
	randomize_weights(hidden_output_link);

	// print layers

// 	printf("layers:\n");
// 	putchar('\t'); print_neural_layer(input_layer);
// 	putchar('\t'); print_neural_layer(hidden_layer);
// 	putchar('\t'); print_neural_layer(output_layer);

	// feed forward

	double input[INPUT_SIZE] = {};
	rand_data(input, INPUT_SIZE);

	printf("### feeding forward input layer... ###\n\n");
	set_neural_active_state(input_layer, input, INPUT_SIZE);
	print_neural_layer(input_layer);

	printf("### feeding forward hidden layer... ###\n\n");
	feed_forward_link(input_hidden_link);
	print_neural_layer(hidden_layer);

	printf("### feeding forward output layer... ###\n\n");
	feed_forward_link(hidden_output_link);
	print_neural_layer(output_layer);

	printf("### feeding backward output layer... ###\n\n");
	feed_backward_layer(output_layer);
	print_neural_layer(output_layer);

	printf("### feeding backward hidden layer... ###\n\n");
	feed_backward_layer(hidden_layer);
	print_neural_layer(hidden_layer);

	// clean up

	destroy_neural_link(input_hidden_link);
	destroy_neural_link(hidden_output_link);

	destroy_neural_layer(input_layer);
	destroy_neural_layer(hidden_layer);
	destroy_neural_layer(output_layer);
}
