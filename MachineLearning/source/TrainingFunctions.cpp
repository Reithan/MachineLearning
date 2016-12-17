#include "pch.h"
#include "TrainingFunctions.h"

namespace ArrayFireTrainer
{

af::array InitializeWeights(dim_t num_elements, const af::dtype type, double min, double max)
{
	af::array order, weights = min + (max - min) * af::range(num_elements, 1, 1, 1, 0, type) / num_elements;
	af::sort(order, weights, af::randu(num_elements), weights);
	return weights;
}

af::array RandomPermutation(dim_t number, dim_t max_value)
{
	double window = (max_value) / (double)number;
	af::array idx = af::seq(0, (double)max_value - 1.0, window);
	idx += af::randu(idx.elements(), u32) % window;
	af::array idx_out;
	af::sort(idx_out, af::array(), idx.as(u32), af::randu(idx.elements()));
	return idx_out;
}

bool CheckGradient(NeuralNet &network, af::dim4 &input_size, dim_t num_classes, int checks, const af::dtype type)
{
	std::cout << "Checking gradient functions.\n";
	double diff, epsilon, tolerance;
	switch (type)
	{
	case f32:
		// values generated via trial and error
		epsilon = 3.25e-2;
		tolerance = 5e-4;
		break;
	case f64:
	default:
		epsilon = 1e-6;
		tolerance = 1e-6;
		break;
	}
	af::array inputs = 2 * af::randu(100, input_size[1], input_size[2], input_size[3], type) - 1;
	auto sizes = network.GetLayerSizes();
	dim_t elements = 0;
	for (auto iter = sizes.begin(); iter != sizes.end(); ++iter)
	{
		elements += iter->elements();
	}
	af::array theta = InitializeWeights(elements, type);

	dim_t num_examples = 100;
	af::array truth = af::randu(num_examples,u8) % num_classes;
	af::array truth_bool = af::constant(0, num_examples, num_classes, b8);
	truth_bool(af::range(num_examples) + truth * num_examples) = 1;
	
	auto output = network.ForwardPropagate(inputs, theta);
	double cost = network.CalculateCost(output, truth_bool, theta);
	double cost_neg, cost_pos, num_grad;
	af::array gradient = network.BackPropagate(output, truth_bool, theta);
	af::array test_weights = theta;
	af::array numeric_gradient(theta.dims(), type);
	af::array checked_gradient(theta.dims(), type);
	numeric_gradient = 0.0;
	checked_gradient = 0.0;
	srand((unsigned int)time(nullptr));
	for (int i = 0; i < checks; i++)
	{
		int j = rand() % elements;
		checked_gradient(j) = gradient(j);
		test_weights(j) = theta(j) + epsilon;
		cost_pos = network.CalculateCost(network.ForwardPropagate(inputs, test_weights), truth_bool, test_weights);
		test_weights(j) = theta(j) - epsilon;
		cost_neg = network.CalculateCost(network.ForwardPropagate(inputs, test_weights), truth_bool, test_weights);
		test_weights(j) = theta(j);
		num_grad = (cost_pos - cost_neg) / (2 * epsilon);
		std::cout << "Check:\t" << std::setw(6) << std::setfill(' ') << i+1 << '\\' << checks << "\t"
			<< j+1 << '\\' << elements << "\t("
			<< std::setw(20) << std::setprecision(16) << std::fixed
			<< num_grad << " : " << gradient(j).scalar<double>() << ")\n";
		numeric_gradient(j) = num_grad;
	}
	diff = af::norm(numeric_gradient - checked_gradient) / af::norm(numeric_gradient + checked_gradient);
	//af_print(af::join(1, numeric_gradient, checked_gradient)(af::randu(10,1,u8),af::span),24);
	std::cout << "Gradient variance:\t" << diff << std::endl;
	std::cout << "Gradient max difference:\t" << af::max<double>( af::abs( numeric_gradient - checked_gradient) ) << std::endl;
	std::cout << "Gradient total difference:\t" << af::sum<double>( af::abs( numeric_gradient - checked_gradient) ) << std::endl;
	std::cout << "Gradient mean difference:\t" << af::sum<double>( af::abs( numeric_gradient - checked_gradient) / checks ) << std::endl;
	
	return (diff < tolerance);
}

void MiniBatchGradientDescent(double alpha, double momentum, dim_t batch_size, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights)
{
	af::dtype type = training_set.type();

	// helper lambda to calculate smoothed mean values
	auto smooth_mean = [](const af::array &values)->double
	{
		double median = af::median<double>(values);
		auto diffs = values - median + 1;
		return af::mean<double>((af::pow(af::abs(diffs), 0.75) - 1) * (2.0 * (diffs > 0) - 1.0) + median);
	};

	const double target = 0.05, minimum = 0.03;
	double current_momentum = 0.0;
	auto num_examples = training_set.dims()[0];
	af::array output, gradient, examples, labels;
	af::array velocity(weights.dims(), type);
	velocity = 0;

	// ensure batch size is <= num_examples
	batch_size = MIN(batch_size, num_examples);

	unsigned int iteration = 0, start = 0, epoch = 0, tally_window = (unsigned int)MAX(10, num_examples / (2 * batch_size));
	std::string header = "Epoch\tIteration Alpha\t\tCost\t\tMoving Avg\n";
	std::cout << header;

	af::array tally(tally_window, type);
	af::seq idx;
	
	af::array order, example_shuffle, label_shuffle;
	af::sort(example_shuffle,order,af::randu(num_examples));
	example_shuffle = training_set(order, af_span);
	label_shuffle = truth(order, af::span);
	
	double time = 0.0,
		mean_check = 0.0,
		mean = 10,
		cost = 0,
		first_window,
		second_window;
	tally = cost;
	double stable = 0.0;
	unsigned short prints = 0;

	while (alpha > 1e-6 && stable < 10 && mean > 1e-6)
	{
		auto timer = af::timer::start();
		start = (iteration * (unsigned int)batch_size) % num_examples;
		idx = af::seq(start, MIN(num_examples - 1, start + (double)batch_size - 1));
		examples = example_shuffle(idx, af::span, af::span, af::span);
		labels = label_shuffle(idx, af::span, af::span, af::span);
		output = network.ForwardPropagate(examples, weights);
		cost = network.CalculateCost(output, labels, weights);
		if (isnan(cost))
		{
			std::cout << "NAN Value detected! Ending gradient descent!\n";
			return;
		}
		gradient = network.BackPropagate(output, labels, weights);
		velocity = velocity * current_momentum + gradient * alpha;
		weights -= velocity;
		tally = af::join(0, tally(af::seq(1, af::end)), af::constant(cost, 1));
		time += timer.stop();
		if (time > 1)
		{
			time = 0.0;
			mean = smooth_mean(tally);
			std::cout << epoch+1 << '\t' << iteration+1 << '\t'
				<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
				<< alpha << '\t' << cost << '\t' << mean << '(' << mean_check << ')' <<std::endl;
			if (0 == ++prints % 20)
			{
				prints = 0;
				std::cout << header;
			}
		}

		if (++iteration == tally_window)
		{
			second_window = smooth_mean(tally);
			current_momentum = momentum;
		}
		else if (0 == iteration % tally_window)
		{
			first_window = second_window;
			second_window = smooth_mean(tally);

			mean_check = (first_window - second_window) / first_window;

			if(mean_check < target)
			{
				static const double epsilon = 1e-6;
			
				// if we're inside minimum, increment stable count
				// if we're above negative minimum, increment by 1/2
				stable = (std::abs(mean_check) < minimum) ? stable + 1.0 : ((mean_check > -minimum) ? stable + 0.5 : 0.0);
				if (mean_check < -minimum || (stable > 1 && stable < 2))
				{
					alpha /= 3;
					velocity /= 3;
				}
				else if(1 - alpha > epsilon && stable > 2)
				{
					alpha = MIN(1, alpha * 2);
					velocity *= 2;
				}
			}
			else
			{
				stable = 0.0f;
			}
		}

		// manually invoke garbage collector to clean up any leftover temporaries
		//	shouldn't be needed long-term.
		af::deviceGC();

		if (idx.s.end == num_examples - 1)
		{
			af::sort(example_shuffle,order,af::randu(num_examples));
			example_shuffle = training_set(order, af_span);
			label_shuffle = truth(order, af::span);
			++epoch;
		}
	}
}

bool SingleExtendedGradientDescent(double &alpha, double &momentum, dim_t batch_size, unsigned int &iteration, unsigned int &epoch, af::array &tally, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights)
{
	af::dtype type = training_set.type();

	// helper lambda to calculate smoothed mean values
	auto smooth_mean = [](const af::array &values)->double
	{
		double median = af::median<double>(values);
		auto diffs = values - median + 1;
		return af::mean<double>((af::pow(af::abs(diffs), 0.75) - 1) * (2.0 * (diffs > 0) - 1.0) + median);
	};

	const double target = 0.05, minimum = 0.03;
	static double current_momentum = 0.0;
	auto num_examples = training_set.dims()[0];
	af::array output, gradient, examples, labels;
	static af::array velocity = af::constant(0.0, weights.dims(), type);

	// ensure batch size is <= num_examples
	batch_size = MIN(batch_size, num_examples);

	unsigned int start = 0, tally_window = 40;//(unsigned int)MAX(10, num_examples / (2 * batch_size));
	std::string header = "Epoch\tIteration Alpha\t\tCost\t\tMoving Avg\n";
	std::cout << header;

	if (tally.elements() == 0)
	{
		tally = af::constant(0.0, tally_window, type);
		tally = 0.0;
	}
	af::seq idx;

	static double time = 0.0,
		stable = 0.0,
		mean_check = 0.0,
		mean = 10,
		cost = 0,
		first_window,
		second_window;
	unsigned short prints = 0;
	unsigned int start_epoch = epoch;

	af::array order, example_shuffle, label_shuffle;
	af::sort(example_shuffle, order, af::randu(num_examples));
	example_shuffle = training_set(order, af_span);
	label_shuffle = truth(order, af::span);

	while (alpha > 1e-6 && stable < 10 && mean > 1e-6)
	{
		auto timer = af::timer::start();
		start = (iteration * (unsigned int)batch_size) % num_examples;
		idx = af::seq(start, MIN(num_examples - 1, start + (double)batch_size - 1));
		examples = example_shuffle(idx, af::span, af::span, af::span);
		labels = label_shuffle(idx, af::span, af::span, af::span);
		output = network.ForwardPropagate(examples, weights);
		cost = network.CalculateCost(output, labels, weights);
		if (isnan(cost))
		{
			std::cout << "NAN Value detected! Ending gradient descent!\n";
			return false;
		}
		gradient = network.BackPropagate(output, labels, weights);
		velocity = velocity * current_momentum + gradient * alpha;
		weights -= velocity;
		tally = af::join(0, tally(af::seq(1, af::end)), af::constant(cost, 1));
		time += timer.stop();
		if (time > 1)
		{
			time = 0.0;
			mean = smooth_mean(tally);
			std::cout << epoch + 1 << '\t' << iteration + 1 << '\t'
				<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
				<< alpha << '\t' << cost << '\t' << mean << '(' << mean_check << ')' << std::endl;
			if (0 == ++prints % 20)
			{
				prints = 0;
				std::cout << header;
			}
		}

		if (++iteration == tally_window)
		{
			second_window = smooth_mean(tally);
			current_momentum = momentum;
		}
		else if (0 == iteration % tally_window)
		{
			first_window = second_window;
			second_window = smooth_mean(tally);

			mean_check = (first_window - second_window) / first_window;

			if (mean_check < target)
			{
				static const double epsilon = 1e-6;

				// if we're inside minimum, increment stable count
				// if we're above negative minimum, increment by 1/2
				stable = (std::abs(mean_check) < minimum) ? stable + 1.0 : ((mean_check > -minimum) ? stable + 0.5 : 0.0);
				if (mean_check < -minimum || (stable > 1 && stable < 2))
				{
					alpha /= 3;
					velocity /= 3;
				}
				else if (1 - alpha > epsilon && stable > 2)
				{
					alpha = MIN(1, alpha * 2);
					velocity *= 2;
				}
			}
			else
			{
				stable = 0.0f;
			}
		}

		// manually invoke garbage collector to clean up any leftover temporaries
		//	shouldn't be needed long-term.
		af::deviceGC();

		if (idx.s.end == num_examples - 1)
		{
			af::sort(example_shuffle, order, af::randu(num_examples));
			example_shuffle = training_set(order, af_span);
			label_shuffle = truth(order, af::span);
			
			af::deviceGC();
			if (++epoch == start_epoch + /*10*/5)
			{
				return true;
			}
		}
	}

	return true;
}

void StochasticGradientDescent(double alpha, double momentum, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights)
{
	MiniBatchGradientDescent(alpha, momentum, 1, network, training_set, truth, weights);
}

void BatchGradientDescent(double alpha, NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights)
{
	MiniBatchGradientDescent(alpha, 0.0, training_set.dims()[0], network, training_set, truth, weights);
}

void ConjugateGradientDescent(NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights)
{
	/* Algorithm pseudo-code taken from
		An Introduction to
		the Conjugate Gradient Method
		Without the Agonizing Pain
		Edition 1 1/4
		Jonathan Richard Shewchuk
		August 4, 1994
		https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
	*/

	// helper lambdas
	auto calculate_gradient = [&](const af::array &inputs)->af::array{
		return network.BackPropagate(network.ForwardPropagate(training_set, inputs), truth, inputs);
	};

	auto inner_product = [](const af::array &lhs, const af::array &rhs)->double{
		return af::sum<double>(af::matmulTN(lhs, rhs));
	};

	auto weights_size = network.GetWeightsSize();
	af::array saved_weights;

	af::array s, r, d;
	double time = 0.0, delta0, alpha, beta, eta, eta_prev, sq_motion, epsilon = 1e-6, sigma = 1e-1;
	bool line_fail;
	short fail_count = 0;

	r = -calculate_gradient(weights);
	d = r;
	delta0 = inner_product(epsilon * r, epsilon * r);

	for (dim_t i = 0, k = 0; i < weights_size && inner_product(r, r) > delta0; ++i)
	{
		auto timer = af::timer::start();
		// Secant line search
		/** start line search **/
		line_fail = false;
		saved_weights = weights;
		size_t j = 0;
		double delta_d = inner_product(d, d);
		alpha = -sigma;
		sq_motion = 0;
		//alpha = -sqrt(1e-2 / delta_d);
		eta_prev = inner_product(calculate_gradient(weights + sigma * d), d);
		do
		{
			eta = inner_product(calculate_gradient(weights), d);
			alpha = alpha * (eta / (eta_prev - eta));
			if (isnan(alpha) || isinf(alpha))
			{
				weights = saved_weights;
				line_fail = true;
				break;
			}
			sq_motion += delta_d * alpha * alpha;
			weights += alpha * d;
			eta_prev = eta;
			++j;
		} while (j < 20 && alpha*alpha*delta_d > epsilon);
		
		// if line-search fails, or doesn't produce movement more than once, we're done.
		if (line_fail || sq_motion < epsilon)
		{
			if (++fail_count == 2)
				break;
		}
		else
			fail_count = 0;
		/** stop line search **/

		// assure cost descent
		s = r;
		r = -calculate_gradient(weights);
		beta = inner_product(r, r - s) / inner_product(s, s); // Polak-Ribier conjugate
		++k;
		// Powell-Beale Restart: http://matlab.izmiran.ru/help/toolbox/nnet/
		if (k == weights_size || beta <= 0 || inner_product(s,r) >= 0.2 * inner_product(r,r))
		{
			d = r;
			k = 0;
		}
		else
		{
			d = r + beta * d;
		}
		time += timer.stop();
		if (time >= 1.0)
		{
			time = 0.0;
			std::cout << "Cost at iteration: "
				<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
				<< i << " - " << network.CalculateCost(network.ForwardPropagate(training_set, weights), truth, weights) << std::endl;
		}

		// manually invoke garbage collector to clean up any leftover temporaries
		//	shouldn't be needed long-term.
		af::deviceGC();
	}
	std::cout << "Cost at cgd end: "
		<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
		<< network.CalculateCost(network.ForwardPropagate(training_set, weights), truth, weights) << std::endl;
}

}
