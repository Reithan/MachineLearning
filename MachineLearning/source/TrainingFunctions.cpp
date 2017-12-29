#include "pch.h"
#include "TrainingFunctions.h"

namespace UAFML
{

namespace CGDFunc
{
	// helper macros
#define CALC_GRAD(inputs) network.BackPropagate(network.ForwardPropagate(training_set, (inputs)), truth, (inputs))

#define CALC_COST(inputs) network.CalculateCost(network.ForwardPropagate(training_set, (inputs)), truth, (inputs))

#define INNER_PROD(lhs, rhs) af::sum<double>(af::matmulTN((lhs), (rhs)))

	typedef bool(*LineSearch)(NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights, af::array &search_direction);

	bool SecantLineSearch(NeuralNet &network, const af::array &training_set, const af::array &truth, af::array &weights, af::array &search_direction)
	{
		/* Algorithm pseudo-code basis from
		An Introduction to
		the Conjugate Gradient Method
		Without the Agonizing Pain
		Edition 1 1/4
		Jonathan Richard Shewchuk
		August 4, 1994
		https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
		*/

		const double epsilon = 1e-6, sigma = 1e-1;
		af::array saved_weights = weights;
		size_t j = 0;
		double
			 delta_d	= INNER_PROD(search_direction, search_direction)
			,alpha		= -sigma
			,eta		= 0.0
			,eta_prev	= 0.0
			,motion		= 0.0
		;
		eta_prev = INNER_PROD(CALC_GRAD(weights + sigma * search_direction), search_direction);
		bool line_fail = false;
		do
		{
			eta = INNER_PROD(CALC_GRAD(weights), search_direction);
			alpha = alpha * eta / (eta_prev - eta);
			if (isnan(alpha) || isinf(alpha))
				return false;
			motion += delta_d * alpha;
			weights += alpha * search_direction;
			eta_prev = eta;
			++j;
		} while (j < 20 && alpha * alpha * delta_d > epsilon * epsilon);

		return (motion > epsilon * epsilon);
	}
}

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
	af::sort(af::array(), idx_out, af::randu(idx.elements()), idx.as(u32));
	return idx_out;
}

bool CheckGradient(NeuralNet &network, af::dim4 &input_size, dim_t num_classes, int checks, const af::dtype type)
{
	std::cout << "Checking gradient functions.\n";
	double diff, epsilon, tolerance;
	switch (type)
	{
	//case b8:
	//case u8:
	//	epsilon = 3.25e-2;
	//	tolerance = 5e-4;
	//	break;
	//case s32:
	//case u32:
	//	epsilon = 3.25e-2;
	//	tolerance = 5e-4;
	//	break;
	//case s64:
	//case u64:
	//	epsilon = 1e-6;
	//	tolerance = 1e-6;
	//	break;
	case f32:
	case c32:
		// values generated via trial and error
		epsilon = 3.25e-2;
		tolerance = 5e-4;
		break;
	case f64:
	case c64:
	default:
		epsilon = 1e-6;
		tolerance = 1e-6;
		break;
	}
	
	af::array inputs = 2 * af::randu(input_size[0], input_size[1], input_size[2], input_size[3], type) - 1;
	auto sizes = network.GetLayerSizes();
	dim_t elements = 0;
	for (auto iter = sizes.begin(); iter != sizes.end(); ++iter)
	{
		elements += iter->elements();
	}
	af::array theta = InitializeWeights(elements, type);

	dim_t num_examples = input_size[0];
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
	af::array selection = RandomPermutation(checks, elements);
	for (int i = 0; i < checks; i++)
	{
		int j = selection(i).scalar<int>();
		checked_gradient(j) = gradient(j);
		test_weights(j) = theta(j) + epsilon;
		cost_pos = network.CalculateCost(network.ForwardPropagate(inputs, test_weights), truth_bool, test_weights);
		test_weights(j) = theta(j) - epsilon;
		cost_neg = network.CalculateCost(network.ForwardPropagate(inputs, test_weights), truth_bool, test_weights);
		test_weights(j) = theta(j);
		num_grad = (cost_pos - cost_neg) / (2 * epsilon);
		std::cout << "Check:\t" << std::setw(6) << std::setfill(' ') << i + 1 << '/' << checks << "\t"
			<< j + 1 << '/' << elements << "\t("
			<< std::setw(20) << std::setprecision(16) << std::fixed
			<< num_grad << " : ";
		switch (type)
		{
		//case s32:
		//case s64:
		//	std::cout << gradient(j).scalar<int>() << ")\n";
		//	break;
		//case u32:
		//case u64:
		//	std::cout << gradient(j).scalar<unsigned int>() << ")\n";
		//	break;
		//case b8:
		//	std::cout << gradient(j).scalar<char>() << ")\n";
		//	break;
		//case u8:
		//	std::cout << gradient(j).scalar<unsigned char>() << ")\n";
		//	break;
		case f32:
		case c32:
			std::cout << gradient(j).scalar<float>() << ")\n";
			break;
		case f64:
		case c64:
		default:
			std::cout << gradient(j).scalar<double>() << ")\n";
			break;
		}
		numeric_gradient(j) = num_grad;
	}
	diff = af::norm(numeric_gradient - checked_gradient) / af::norm(numeric_gradient + checked_gradient);
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
	//std::cout << header;

	if (tally.elements() == 0)
	{
		tally = af::constant(0.0, tally_window, type);
		tally = 0.0;
	}
	af::seq idx;

	static double time = 0.0,
		//stable = 0.0,
		mean_check = 0.0,
		mean = 10,
		cost = 0,
		first_window,
		second_window;
	static unsigned short prints = 0;
	unsigned int start_epoch = epoch;

	af::array order, example_shuffle, label_shuffle;
	af::sort(example_shuffle, order, af::randu(num_examples));
	example_shuffle = training_set(order, af_span);
	label_shuffle = truth(order, af::span);

	if (alpha > 1e-6 && /*stable < 10 &&*/ mean > 1e-6)
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
			if (0 == prints++)
				std::cout << header;
			else if (prints == 20)
				prints = 0;
			mean = smooth_mean(tally);
			std::cout << epoch + 1 << '\t' << iteration + 1 << '\t'
				<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
				<< alpha << '\t' << cost << '\t' << mean << '(' << mean_check << ')' << std::endl;
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
				//stable = (std::abs(mean_check) < minimum) ? stable + 1.0 : ((mean_check > -minimum) ? stable + 0.5 : 0.0);
				if (mean_check < -minimum )//|| (stable > 1.0 && stable < 2.0))
				{
					alpha /= 3;
					velocity /= 3;
				}
				else if (1 - alpha > epsilon)// && stable > 2.0)
				{
					alpha = MIN(1, alpha * 2);
					velocity *= 2;
				}
			}
			//else
			//{
			//	stable = 0.0f;
			//}
		}

		// manually invoke garbage collector to clean up any leftover temporaries
		//	shouldn't be needed long-term.
		af::deviceGC();

		++epoch;
		return true;
	}
	return false;
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
	/* Algorithm pseudo-code basis from
		An Introduction to
		the Conjugate Gradient Method
		Without the Agonizing Pain
		Edition 1 1/4
		Jonathan Richard Shewchuk
		August 4, 1994
		https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
	*/

	auto weights_size = network.GetWeightsSize();

	CGDFunc::LineSearch line_search = &CGDFunc::SecantLineSearch;
	
	af::array residual, prev_residual, search_direction, delta_residual, prev_delta_residual;
	double time = 0.0, residual_sq_mag,
			beta, epsilon = 1e-6;
	short fail_count = 0;

	residual = -CALC_GRAD(weights);
	search_direction = residual;
	residual_sq_mag = INNER_PROD(residual, search_direction);
	delta_residual = af::constant(0.0, weights.dims());

	for (dim_t i = 0, k = 0; i < weights_size && INNER_PROD(residual, search_direction) > epsilon * epsilon * residual_sq_mag; ++i)
	{
		auto timer = af::timer::start();

		af::array saved_weights = weights;
		// if line-search fails, or doesn't produce movement more than once, we're done.
		if (!line_search(network, training_set, truth, weights, search_direction))
		{
			weights = saved_weights + epsilon * search_direction;
			if (++fail_count == 2)
				break;
		}
		else
			fail_count = 0;

		prev_residual = residual;
		residual = -CALC_GRAD(weights);
		prev_delta_residual = delta_residual;
		delta_residual = residual - prev_residual;
		beta = INNER_PROD(prev_delta_residual, residual) / INNER_PROD(prev_residual, prev_residual); // Polak-Ribier conjugate (corrected)

		++k;
		// Powell-Beale Restart: http://matlab.izmiran.ru/help/toolbox/nnet/
		if (k == weights_size || beta <= 0 || INNER_PROD(prev_residual, residual) >= 0.2 * INNER_PROD(residual,residual))
		{
			search_direction = residual;
			k = 0;
		}
		else
		{
			search_direction = residual + beta * search_direction;
		}
		time += timer.stop();
		if (time >= 1.0)
		{
			time = 0.0;
			std::cout << "Cost at iteration: "
				<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
				<< i << " - " << CALC_COST(weights) << std::endl;
		}

		// manually invoke garbage collector to clean up any leftover temporaries
		//	shouldn't be needed long-term.
		af::deviceGC();
	}
	std::cout << "Cost at cgd end: "
		<< std::setw(10) << std::setfill(' ') << std::setprecision(6) << std::fixed
		<< CALC_COST(weights) << std::endl;
}

}
