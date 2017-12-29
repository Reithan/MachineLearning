#include "pch.h"
#include "MaxPoolLayer.h"

namespace UAFML
{

bool MaxPoolLayer::ForwardPropagate(af::array &values, const af::array &/*weights*/)
{
	auto value_size = values.dims();
	values = af::reorder(values,2,3,0,1);
	values = af::unwrap(values,2,2,2,2);
	af::max(values, _key, values, 0);
	values = af::moddims(values,value_size[2]/2,value_size[3]/2,value_size[0],value_size[1]);
	values = af::reorder(values,2,3,0,1);

	return true;
}

af::array MaxPoolLayer::BackPropagate(af::array &error, const af::array &/*weights*/)
{
	auto error_size = error.dims();

	error = af::reorder(error,2,3,0,1);
	af::array output = af::constant(0, 4, error_size[2] * error_size[3], error_size[0], error_size[1], error.type());
	auto idx = af::flat(_key) + af::range(error_size.elements()) * 4;
	output(idx) = af::flat(error);
	error = af::wrap(output,error_size[2] * 2, error_size[3] * 2,2,2,2,2);
	error = af::reorder(error,2,3,0,1);
	
	return af::array();
}

double MaxPoolLayer::CalculateCost(const af::array &output, const af::array &truth)
{
	return af::sum<double>(af::pow(output - truth, 2)) / (2 * output.dims()[0]);
}

af::array MaxPoolLayer::CalculateError(af::array &values, const af::array &truth, const af::array &weights)
{
	values = (values - truth) / values.dims()[0];
	return BackPropagate(values, weights);
}

}
