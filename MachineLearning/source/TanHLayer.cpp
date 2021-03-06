#include "pch.h"
#include "TanHLayer.h"

namespace UAFML
{

bool TanHLayer::ForwardPropagate(af::array &values, const af::array &/*weights*/)
{
	_activation = values = af::tanh(values);
	return true;
}

af::array TanHLayer::BackPropagate(af::array &error, const af::array &/*weights*/)
{
	error = error * (1 - af::pow(_activation,2));
	return af::array();
}

double TanHLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>(af::pow(output - truth, 2)) / (2 * output.dims(0));
}

af::array TanHLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims(0);
	return BackPropagate(values, weights);
}

}
