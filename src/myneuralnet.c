#include <stdio.h>
#include <time.h>

#include "neural_network.h"

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 8

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

	neural_layer *input_layer = create_neural_layer(INPUT_SIZE, "input layer");
	neural_layer *hidden_layer = create_neural_layer(HIDDEN_SIZE, "hidden layer");
	neural_layer *output_layer = create_neural_layer(OUTPUT_SIZE, "output layer");

	// link layers

	neural_link *input_hidden_link = link_neural_layers(input_layer, hidden_layer);
	neural_link *hidden_output_link = link_neural_layers(hidden_layer, output_layer);

	// randomize link weights

	randomize_weights(input_hidden_link);
	randomize_weights(hidden_output_link);

	// print layers

	printf("layers:\n");
	putchar('\t'); print_neural_layer(input_layer);
	putchar('\t'); print_neural_layer(hidden_layer);
	putchar('\t'); print_neural_layer(output_layer);

	// feed forward

	double input[INPUT_SIZE] = {};
	double hidden[HIDDEN_SIZE] = {};
	double output[OUTPUT_SIZE] = {};

	rand_data(input, INPUT_SIZE);
	printf("\ninput "); print_data(input, INPUT_SIZE);

	feed_forward(input_hidden_link, input, hidden);
	printf("\nhidden "); print_data(hidden, HIDDEN_SIZE);

	feed_forward(hidden_output_link, hidden, output);
	printf("\noutput "); print_data(output, OUTPUT_SIZE);
}
