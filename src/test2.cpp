#include "c10/core/ScalarType.h"
#include <torch/all.h>
//#include "../../nlib/include/nlib/nl_utils.h"

using namespace torch;
using namespace std;

static Tensor toComplex = torch::cat ({torch::complex (torch::tensor ({1}, kFloat), torch::tensor ({0}, kFloat)),
										  torch::complex (torch::tensor ({0}, kFloat), torch::tensor ({1}, kFloat))});

Tensor act (const Tensor &cc, const Tensor &other) {
	return cc;
}

#define PC(tt) {cout << #tt << "\nreal:\n" << real(tt) << "\nimag:\n" << imag(tt) << endl;}

using cpx = std::complex<double>;

int main () {
	Tensor bubu = torch::tensor ({1,2}, kComplexFloat);
	Tensor vects = torch::tensor ({{1,2},{3,4}}, kComplexFloat);

	Tensor ccbubu = toComplex.dot (bubu);
	Tensor ccvects = toComplex.unsqueeze (0).mm(vects.t()).squeeze ();
	Tensor ccprod = ccbubu * ccvects;

	//cout << real(cc) << endl << imag (cc) << endl;
	//PC(cc);
	PC(ccbubu);
	PC(ccvects);
	PC(ccprod);
	cout << torch::stack ({real (ccprod), imag(ccprod)},1) << endl;
	cout << torch::stack ({real (ccprod), imag(ccprod)}, 1)[0] << endl;


	cpx cc1 = 1. + 2i;
	cpx v11 = 1. + 2i;
	cpx v12 = 3. + 4i;


	cout << "cc1 v11\n" << cc1 * v11 << endl;
	cout << "cc1 v12\n" << cc1 * v12 << endl;


	//cout << act (cc, vects) << endl;
}
