#include "pch.h"
#include "SigmoidLayer.h"

namespace ArrayFireTrainer
{

bool SigmoidLayer::ForwardPropagate(af::array &values, const af::array &/*weights*/)
{
	_activation = values = af::sigmoid(values);
	return true;
}

af::array SigmoidLayer::BackPropagate(af::array &error, const af::array &/*weights*/)
{
	error = error * _activation * (1 - _activation);
	return af::array();
}

double SigmoidLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>((-truth) * af::log(output) - (1 - truth) * af::log(1 - output)) / output.dims()[0];
}

af::array SigmoidLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims()[0];
	return af::array();
}

}
