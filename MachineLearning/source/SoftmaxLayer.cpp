#include "pch.h"
#include "SoftmaxLayer.h"

namespace ArrayFireTrainer
{

bool SoftmaxLayer::ForwardPropagate(af::array &values, const af::array &/*weights*/)
{
	af::array exp_scores = af::exp(values);
	af::array denoms = 1.0 / af::sum(exp_scores,1);
	_activation = values = af::batchFunc(exp_scores, denoms,
		[](const af::array &lhs, const af::array &rhs)->af::array{
			return lhs * rhs;
		});
	return true;
}

af::array SoftmaxLayer::BackPropagate(af::array &error, const af::array &/*weights*/)
{
	return af::array();
}

double SoftmaxLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	af::array correct_logprobs = -truth * af::log(output);	
	double cost = af::sum<double>(correct_logprobs) / output.dims()[0];

	if(isnan(cost))
	{
		af::replace(correct_logprobs, af::isNaN(correct_logprobs) || af::isInf(correct_logprobs), 100.0);
		cost = af::sum<double>(correct_logprobs) / output.dims()[0];
	}

	return cost;
}

af::array SoftmaxLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims()[0];
	return af::array();
}

}
