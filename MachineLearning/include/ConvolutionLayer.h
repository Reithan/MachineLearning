#pragma once

#include "INetworkLayer.h"

namespace ArrayFireTrainer
{

class ConvolutionLayer : public INetworkLayer
{
public:
	ConvolutionLayer(const af::dim4 &size) : INetworkLayer(size) {}
	virtual ~ConvolutionLayer();

	virtual bool ForwardPropagate(af::array &values, const af::array &weights);
	virtual af::array BackPropagate(af::array &error, const af::array &weights);

	virtual double CalculateCost(const af::array &output, const af::array &truth);
	virtual af::array CalculateError(af::array &values, const af::array &truth, const af::array &weights);
protected:
	af::array _input;
};

}
