#pragma once

#include "INetworkLayer.h"

namespace UAFML
{

class DenseLayer : public INetworkLayer
{
public:
	DenseLayer(const af::dim4 &size) : INetworkLayer(size) {}
	virtual ~DenseLayer();

	virtual bool ForwardPropagate(af::array &values, const af::array &weights);
	virtual af::array BackPropagate(af::array &error, const af::array &weights);

	virtual double CalculateCost(const af::array &output, const af::array &truth);
	virtual af::array CalculateError(af::array &values, const af::array &truth, const af::array &weights);
private:
	af::array _input;
	af::dim4 _input_size;
};

}
