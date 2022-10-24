#include <torch/all.h>
#include "../../nlib/include/nlib/nl_utils.h"
#include "lietorch/pose.h"
#include "lietorch/unit_complex.h"

using namespace torch;
using namespace std;
using namespace lietorch;


int main ()
{
	UnitComplex ori(M_PI/6);
	UnitComplex ori2(M_PI/4);
	Tensor point = torch::tensor ({{1, 2},{3,4}}, kFloat);
	Tensor point2 = torch::tensor ({3,4}, kFloat);
	Tensor point3 = torch::tensor ({5,6}, kFloat);
	Pose2 poseTest(point2, ori);

	cout << poseTest.inverse () * point3 << endl;


	std::complex<double> oristd = exp(M_PI/6 * 1i);
	std::complex<double> ori2std = exp(M_PI/4 * 1i);
	std::complex<double> point3std = 5. + 6i;
	std::complex<double> pointstd = 1. + 2i;
	std::complex<double> point2std = 3. + 4i;

	auto orinv = conj(oristd);

	cout << "std\n" << orinv * point3std  + orinv * (- point2std) << endl;
}
