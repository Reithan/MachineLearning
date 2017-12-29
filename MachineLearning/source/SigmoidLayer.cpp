#include "pch.h"
#include "SigmoidLayer.h"

namespace UAFML
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
	auto
		 output_log		= af::log(output)
		,inv_output_log	= af::log(1 - output)
	;
	double cost = af::sum<double>((-truth) * output_log - (1 - truth) * inv_output_log) / output.dims()[0];
	if (isnan(cost) || isinf(cost))
	{
		af::replace(output_log, !(af::isNaN(output_log) || af::isInf(output_log)), -100.0);
		af::replace(inv_output_log, !(af::isNaN(inv_output_log) || af::isInf(inv_output_log)), -100.0);
		cost = af::sum<double>((-truth) * output_log - (1 - truth) * inv_output_log) / output.dims()[0];
	}
	return cost;
}

af::array SigmoidLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims()[0];
	return af::array();
}

}
