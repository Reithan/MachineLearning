#pragma once

#include "INetworkLayer.h"

namespace UAFML
{

class ReLULayer : public INetworkLayer
{
public:
	ReLULayer(const af::dim4 &size) : INetworkLayer(size) {}
	virtual ~ReLULayer() {}

	virtual bool ForwardPropagate(af::array &values, const af::array &weights);
	virtual af::array BackPropagate(af::array &error, const af::array &weights);

	virtual double CalculateCost(const af::array &output, const af::array &truth);
	virtual af::array CalculateError(af::array &values, const af::array &truth, const af::array &weights);
protected:
	af::array _input;
};

}
