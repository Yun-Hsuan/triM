// This is the simple infinite 1D-TEBD source code 
// And the matrices and tensors can be pushed to GPU or CPU by guni10Create()
// , which would detect the usage of the deivces. 
//  
#include <iostream>
#include <assert.h>
#include <map>
using namespace std;
#include "uni10.hpp"
using namespace uni10;
#include <time.h>
#include <stdlib.h>

size_t CHI = 250;

UniTensor S1XXZDE(double Jz, double D, double E, double spin = 1.);
void entangleSpec(Matrix& L, size_t chi);

void entangleSpec(Matrix& L, size_t chi){
  vector<double> xi;
  for(size_t i = 0; i < chi; i++){
    xi.push_back(-log(L[i]*L[i])); 
  }
  fprintf(stderr, "["); 
  for(size_t i = 0; i < chi; i++){
    if(i == 0)
      fprintf(stderr, " %.5f", xi[i]); 
    else
      fprintf(stderr, ", %.5f", xi[i]); 
  }
  fprintf(stderr, "]\n"); 
}

UniTensor S1XXZDE(double Jz, double D, double E, double spin){

  Matrix S1p(3, 3);
  double elemS1[] = {0, sqrt(2), 0, 0, 0, sqrt(2), 0, 0, 0};
  S1p.setElem(elemS1);
  Matrix S1m = S1p;
  //cout << S1p << endl;
  S1m.transpose();
  //cout << S1m << endl;
  Matrix S1z(3, 3);
  double elemS1z[] = {1, 0, 0, 0, 0, 0, 0, 0, -1};
  S1z.setElem(elemS1z);
  Matrix S1id(3, 3);
  S1id.identity();
  //cout << S1id << endl;
  Matrix sqS1z = S1z*S1z;
  Matrix sqS1p = S1p*S1p;
  Matrix sqS1m = S1m*S1m;

  otimes(S1p, S1m);
  Matrix XXterm = 0.5 * (otimes(S1p, S1m) + otimes(S1m, S1p));
  Matrix Jzterm = Jz * (otimes(S1z, S1z));
  Matrix Dterm = D * (otimes(S1id, sqS1z)+otimes(sqS1z, S1id));
  Matrix Eterm = 0.5 * E * (otimes(sqS1p, S1id)+otimes(S1id, sqS1p)+otimes(S1id, sqS1m)+otimes(sqS1m, S1id));
  
  vector<Bond> Hbonds(4, Bond(BD_IN, 3));
  Hbonds [2] = Bond(BD_OUT, 3);
  Hbonds [3] = Hbonds[2];
  UniTensor H(Hbonds);
  //H.printDiagram();
  H.putBlock(XXterm + Jzterm + Dterm + Eterm);
  //cout << H << endl;
  return H;

}

void update(UniTensor& ALa, UniTensor& BLb, map<Qnum, Matrix>& La, map<Qnum, Matrix>& Lb, UniTensor& U, Network& iTEBD, Network& updateA);
double measure(UniTensor& ALa, UniTensor& BLb, map<Qnum, Matrix>& La, map<Qnum, Matrix>& Lb, UniTensor& Op, Network& MPS, Network& meas);
double expectation(UniTensor& L, UniTensor& R, UniTensor& Op, Network& MPS, Network& meas);
double measure2(UniTensor& ALa, UniTensor& BLb, map<Qnum, Matrix>& Lb, UniTensor& expH, Network & iTEBD, double delta);

int main(int argc, char** argv){
  // Define the parameters of the model / simulation
  //
  if(argc == 2)
    guni10Create(atoi(argv[1]));
  else if(argc == 1)
    guni10Create(0);

  double delta = 0.01;
  int N = 10;
  double eps = 1E-9;
  /*** Initialization ***/
  Qnum q0(0);
  double Jz = 1.0;
  double D = 0.0;
  double E = 6.5;
  UniTensor H = S1XXZDE(Jz, D, E);
  vector<Bond> bondH = H.bond();
  UniTensor U(bondH, "U");
  U.putBlock(q0, takeExp(-delta, H.getBlock(q0)));
  //cout << U << endl;
  //exit(0);
  
  Bond bdi_chi(BD_IN, CHI);
  vector<Bond> bond3;
  bond3.push_back(bdi_chi);
  bond3.push_back(bdi_chi);
  bond3.push_back(bondH[2]);

  UniTensor ALa(bond3, "ALa");
  UniTensor BLb(bond3, "BLb");
  map<Qnum, Matrix> La;
  map<Qnum, Matrix> Lb;

  Matrix I_chi(CHI, CHI, true);
  I_chi.randomize();
  La[q0] = I_chi;
  I_chi.randomize();
  Lb[q0] = I_chi;

  ALa.randomize();
  BLb.randomize();

  Network iTEBD("itebd.net");
  Network updateA("updateA.net");
  Network MPS("MPS.net");
  Network meas("measure.net");

  clock_t t;
  t = clock();
  double M0 = 0, M1 = 0;
  for(int step = 0; step < N; step++){
    //printf("step = %d\n", step);
    update(ALa, BLb, La, Lb, U, iTEBD, updateA);
    update(BLb, ALa, Lb, La, U, iTEBD, updateA);
    M1 = measure(ALa, BLb, La, Lb, H, MPS, meas);
    fprintf(stderr, "Eg: %.15f\n", measure(ALa, BLb, La, Lb, H, MPS, meas));
    if(fabs(M1-M0) < eps)
      break;
    M0 = M1;
    //cout<<"E = "<<setprecision(12)<<measure(ALa, BLb, La, Lb, H, MPS, meas)<<endl;
  }
  t = clock() - t;
  printf ("It took %ld clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
  fprintf(stderr, "Eg: %.15f\n", measure(ALa, BLb, La, Lb, H, MPS, meas));
  //entangleSpec(La[q0], CHI);
  //entangleSpec(Lb[q0], CHI);
  iTEBD.profile();
  updateA.profile();
  H.profile();
}


void update(UniTensor& ALa, UniTensor& BLb, map<Qnum, Matrix>& La, map<Qnum, Matrix>& Lb, UniTensor& expH, Network& iTEBD, Network& updateA){
  Qnum q0(0);
  iTEBD.putTensor("ALa", &ALa);
  iTEBD.putTensor("BLb", &BLb);
  iTEBD.putTensor("expH", &expH);
  UniTensor C = iTEBD.launch();
  UniTensor Theta(C.bond(), "Theta");
  Theta.putBlock(q0, Lb[q0] * C.getBlock(q0));
  Theta.permute(2);
  vector<Matrix> rets = Theta.getBlock(q0).svd();
  int dim = CHI < rets[1].row() ? CHI : rets[1].row();
  double norm = rets[1].resize(dim, dim).norm();
  rets[1] *= (1 / norm);
  La[q0] = rets[1];
  Bond bdi(BD_IN, dim);
  vector<Bond> bond3 = ALa.bond();
  bond3[0] = bdi;
  BLb.assign(bond3);
  Matrix blk = BLb.getBlock(q0);
  blk.setElem(rets[2].getElem(), rets[2].isOngpu());
  BLb.putBlock(q0, blk);
  updateA.putTensor("BLb", &BLb);
  updateA.putTensor("C", &C);
  ALa = updateA.launch();
  ALa *= (1 / norm);
}

double measure2(UniTensor& ALa, UniTensor& BLb, map<Qnum, Matrix>& Lb, UniTensor& expH, Network & iTEBD, double delta){
  Qnum q0(0);
  iTEBD.putTensor("ALa", &ALa);
  iTEBD.putTensor("BLb", &BLb);
  iTEBD.putTensor("expH", &expH);
  UniTensor C = iTEBD.launch();
  UniTensor Theta(C.bond(), "Theta");
  Theta.putBlock(q0, Lb[q0] * C.getBlock(q0));
  Theta.permute(2);
  UniTensor Theta2 = Theta;
  UniTensor val = Theta * Theta2;
  return -log(val[0]) / delta / 2;
}

double measure(UniTensor& ALa, UniTensor& BLb, map<Qnum, Matrix>& La, map<Qnum, Matrix>& Lb, UniTensor& Op, Network& MPS, Network& meas){
  Qnum q0(0);
  UniTensor A1(ALa);
  A1.permute(1);
  A1.putBlock(q0, Lb[q0] * A1.getBlock(q0));
  A1.permute(2);

  UniTensor B1(BLb);
  B1.permute(1);
  B1.putBlock(q0, La[q0] * B1.getBlock(q0));
  B1.permute(2);
  double val = expectation(A1, BLb, Op, MPS, meas);
  val += expectation(B1, ALa, Op, MPS, meas);
  return val / 2;
}

double expectation(UniTensor& L, UniTensor& R, UniTensor& Op, Network& MPS, Network& meas){
  Qnum q0(0);
  MPS.putTensor("L", &L);
  MPS.putTensor("R", &R);
  UniTensor psi = MPS.launch();
  double norm = psi.getBlock(q0).norm();
  norm *= norm;
  meas.putTensor("bra", &psi);
  meas.putTensorT("ket", &psi);
  meas.putTensor("Op", &Op);
  return meas.launch()[0] / norm;
}

