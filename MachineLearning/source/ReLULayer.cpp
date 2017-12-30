#include "pch.h"
#include "ReLULayer.h"

namespace UAFML
{

bool ReLULayer::ForwardPropagate(af::array &values, const af::array &/*weights*/)
{
	_input = values;
	af::replace(values, values > 0.0, 0.0);
	return true;
}

af::array ReLULayer::BackPropagate(af::array &error, const af::array &/*weights*/)
{
	af::replace(error, _input > 0.0, 0.0);
	return af::array();
}

double ReLULayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>(af::pow(output - truth, 2)) / (2 * output.dims(0));
}

af::array ReLULayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims(0);
	return BackPropagate(values, weights);
}

}
