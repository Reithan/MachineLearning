#pragma once

#include "ConvolutionLayer.h"

namespace ArrayFireTrainer
{

class PaddedConvolutionLayer : public ConvolutionLayer
{
public:
	PaddedConvolutionLayer(const af::dim4 &size) : ConvolutionLayer(size) {}
	virtual ~PaddedConvolutionLayer();

	virtual bool ForwardPropagate(af::array &values, const af::array &weights);
	virtual af::array BackPropagate(af::array &error, const af::array &weights);
};

}
