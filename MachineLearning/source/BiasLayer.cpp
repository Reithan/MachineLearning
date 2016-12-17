#include "pch.h"
#include "BiasLayer.h"


namespace ArrayFireTrainer
{

bool BiasLayer::ForwardPropagate(af::array &values, const af::array &weights)
{
	values += af::tile(weights, (unsigned int)values.dims()[0]);
	return true;
}

af::array BiasLayer::BackPropagate(af::array &error, const af::array &weights)
{
	return af::sum(error,0);
}


double BiasLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>(af::pow(output - truth, 2)) / (2 * output.dims()[0]);
}

af::array BiasLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims()[0];
	return BackPropagate(values, weights);
}

}
