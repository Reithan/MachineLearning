#pragma once
#include "INetworkLayer.h"

#include <vector>
#include <memory>

namespace ArrayFireTrainer
{

class NeuralNet
{
public:
	NeuralNet(double lambda = 0.0, double dropout = 0.0);
	~NeuralNet();

	void SetLambda	(double value);
	void SetDropout	(double value);
	double GetLambda	() const { return _lambda	; }
	double GetDropout	() const { return _dropout	; }

	template <typename Type>
	size_t AddLayer(const af::dim4 &size);
	std::vector<af::dim4> GetLayerSizes();
	dim_t GetWeightsSize();

	// inputs = dataset, weights = theta
	//		output = final output values
	af::array ForwardPropagate(const af::array &inputs, const af::array &weights);
	// inputs = output from forward prop, true values, weights(theta) used in forward prop
	//		output = gradient of theta
	af::array BackPropagate(const af::array &output, const af::array &truth, const af::array &weights);
	// inputs = output from forward prop, true values, weights(theta) used in forward prop
	//		output = cost for this set of weights against this training set.
	double CalculateCost(const af::array &output, const af::array &truth, const af::array &weights);
	
private:
	double _lambda;
	double _dropout;

	std::vector<std::unique_ptr<INetworkLayer>> _layers;
};

template <typename Type>
size_t NeuralNet::AddLayer(const af::dim4 &size = af::dim4(0,0,0,0))
{
	_layers.emplace_back(std::unique_ptr<INetworkLayer>(new Type(size)));
	return _layers.size();
}

}
