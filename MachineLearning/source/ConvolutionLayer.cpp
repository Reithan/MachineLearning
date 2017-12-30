#include "pch.h"
#include "ConvolutionLayer.h"

namespace UAFML
{

ConvolutionLayer::~ConvolutionLayer()
{
}

bool ConvolutionLayer::ForwardPropagate(af::array &values, const af::array &weights)
{
	_input = values;

	values = af::reorder(values, 2, 3, 1, 0);
	auto filters = af::reorder(weights, 2, 3, 1, 0);
	filters = af::flip(af::flip(af::flip(filters, 0), 1), 2);

	auto filter_dim = filters.dims();
	auto value_dim = values.dims();
	af::dim4 new_val_dim;
	new_val_dim[0] = filter_dim[0] + value_dim[0] - 1;
	new_val_dim[1] = filter_dim[1] + value_dim[1] - 1;
	new_val_dim[2] = filter_dim[3];
	new_val_dim[3] = value_dim[3];
	af::array new_values(new_val_dim, values.type());

	double x_dim = (double)MIN(filter_dim[0], value_dim[0]);
	double y_dim = (double)MIN(filter_dim[1], value_dim[1]);
	double z_dim = (double)MIN(filter_dim[2], value_dim[2]);
	af::seq cutout_x(x_dim - 1, -x_dim);
	af::seq cutout_y(y_dim - 1, -y_dim);
	af::seq cutout_z(z_dim - 1, -z_dim);

	for (int kernel = 0; kernel < filter_dim[3]; ++kernel)
	{
		if(value_dim[2] == 1 && filter_dim[2] == 1) // TODO won't be needed after af issue #1442 is in released build
			new_values(af::span, af::span, kernel, af::span) = af::convolve2(values, filters(af::span, af::span, af::span, kernel), AF_CONV_EXPAND);
		else
			new_values(af::span, af::span, kernel, af::span) = af::convolve3(values, filters(af::span, af::span, af::span, kernel), AF_CONV_EXPAND)(af::span, af::span, cutout_z);

		af::deviceGC();
	}

	values = new_values(cutout_x, cutout_y);
	values = af::reorder(values, 3, 2, 0, 1);

	return true;
}

af::array ConvolutionLayer::BackPropagate(af::array &error, const af::array &weights)
{
	af::array gradient = af::reorder(_input, 2, 3, 1, 0);
	auto filter = af::flip(af::flip(af::reorder(error, 2, 3, 1, 0), 0), 1);
	error = af::reorder(error, 2, 3, 1, 0);

	auto error_dim = error.dims();
	auto gradient_dim = gradient.dims();
	auto weight_dim = weights.dims();
	auto filter_dim = filter.dims();

	af::dim4 new_val_dim;
	new_val_dim[0] = gradient_dim[0] + filter_dim[0] - 1;
	new_val_dim[1] = gradient_dim[1] + filter_dim[1] - 1;
	new_val_dim[2] = gradient_dim[2];
	new_val_dim[3] = filter_dim[2];
	af::array new_values(new_val_dim, gradient.type());

	// gradient
	for (int kernel = 0; kernel < filter_dim[2]; ++kernel)
	{
		if(gradient_dim[2] == 1 || gradient_dim[3] == 1) // TODO won't be needed after af issue #1442 is in released build
			new_values(af::span, af::span, af::span, kernel) = af::sum(af::convolve2(gradient, filter(af::span, af::span, kernel), AF_CONV_EXPAND), 3);
		else
			new_values(af::span, af::span, af::span, kernel) = af::sum(af::convolve3(gradient, filter(af::span, af::span, kernel), AF_CONV_EXPAND), 3);

		af::deviceGC();
	}

	double x_dim = (double)MIN(filter_dim[0], gradient_dim[0]);
	double y_dim = (double)MIN(filter_dim[1], gradient_dim[1]);
	af::seq cutout_x(x_dim - 1, -x_dim);
	af::seq cutout_y(y_dim - 1, -y_dim);

	gradient = af::reorder(new_values(cutout_x, cutout_y), 3, 2, 0, 1);
	gradient_dim = gradient.dims();

	// error
	filter = af::reorder(weights, 2, 3, 1, 0);
	filter_dim = filter.dims();

	new_val_dim[0] = error_dim[0] + filter_dim[0] - 1;
	new_val_dim[1] = error_dim[1] + filter_dim[1] - 1;
	new_val_dim[2] = filter_dim[2];
	new_val_dim[3] = error_dim[3];
	new_values = af::constant(0, new_val_dim, error.type());

	for (int kernel = 0; kernel < filter_dim[3]; ++kernel)
	{
		if (filter_dim[2] == 1 || error_dim[3] == 1) // TODO won't be needed after af issue #1442 is in released build
			new_values += af::convolve2(error(af::span, af::span, kernel), filter(af::span, af::span, af::span, kernel), AF_CONV_EXPAND);
		else
			new_values += af::convolve3(error(af::span, af::span, kernel), filter(af::span, af::span, af::span, kernel), AF_CONV_EXPAND);

		af::deviceGC();
	}
	error = af::reorder(new_values, 3, 2, 0, 1);

	return gradient;
}

double ConvolutionLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>(af::pow(output - truth, 2)) / (2 * output.dims(0));
}

af::array ConvolutionLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims(0);
	return BackPropagate(values, weights);
}

}
