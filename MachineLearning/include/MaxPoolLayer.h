#pragma once

#include "INetworkLayer.h"

namespace UAFML
{

class MaxPoolLayer : public INetworkLayer
{
public:
	MaxPoolLayer(const af::dim4 &) {}
	virtual ~MaxPoolLayer() {}

	virtual bool ForwardPropagate(af::array &values, const af::array &weights);
	virtual af::array BackPropagate(af::array &error, const af::array &weights);

	virtual double CalculateCost(const af::array &output, const af::array &truth);
	virtual af::array CalculateError(af::array &values, const af::array &truth, const af::array &weights);
private:
	af::array _key;
};

}
