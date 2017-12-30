#include "pch.h"
#include "DenseLayer.h"

namespace UAFML
{

DenseLayer::~DenseLayer()
{
}

bool DenseLayer::ForwardPropagate(af::array &values, const af::array &weights)
{
	auto dims = values.dims();
	_input_size = values.dims();
	_input = af::moddims(values, dims[0], dims[1] * dims[2] * dims[3]);
	values = af::matmul(_input, weights);
	return true;
}

af::array DenseLayer::BackPropagate(af::array &error, const af::array &weights)
{
	af::array gradient = af::matmulTN(_input, error);
	error = af::matmulNT(error, weights);
	error = af::moddims(error, _input_size);
	return gradient;
}

double DenseLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>(af::pow(output - truth, 2)) / (2 * output.dims(0));
}

af::array DenseLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims(0);
	return BackPropagate(values, weights);
}

}
