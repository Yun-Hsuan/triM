#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <magma.h>
#include <magma_lapack.h>
#include <magma_types.h>
//#include <magma_v2.h>
#include <uni10/tools/uni10_tools.h>
#include <uni10/tools/uni10_tools_gpu_kernel.h>



namespace uni10{

  void magma_print_matrix(double* v, int m, int n, int ldda){

    double* bufv = (double*)malloc(m*n*sizeof(double));
    printf("[\n");
    if(magma_is_devptr(v) == 1){
        printf("\n-- On GPU --\n\n");
        magma_dgetmatrix(n, m, v, ldda, bufv, n);
    }else if(magma_is_devptr(v) == 0){
        printf("-- On CPU --\n");
        memcpy(bufv, v, m*n*sizeof(double));
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n ; j++)
            printf("%8.5f  ", bufv[i*n+j]);
        printf("\n");
    }
    printf("];\n");

  }

  void cuda_print_matrix(double* v, int m, int n){

    double* bufv = (double*)malloc(m*n*sizeof(double));
    printf("[\n");
    if(magma_is_devptr(v) == 1){
        printf("\n-- On GPU --\n\n");
        cudaMemcpy(bufv, v, m*n*sizeof(double), cudaMemcpyDeviceToHost);
    }else if(magma_is_devptr(v) == 0){
        printf("-- On CPU --\n");
        memcpy(bufv, v, m*n*sizeof(double));
    }

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n ; j++)
            printf("%8.5f  ", bufv[i*n+j]);
        printf("\n");
    }
    printf("];\n");

  }

  void guni10_create(int GMODE_, bool printUsage){
    //define the variable ONGPU and GMODE from .uni10rc  
    //.uni10rc
    //-------------------------------------//
    //DEV: gpu0, gpu1, ...
    //GMODE: 
    //-------------------------------------//
    char RcDir[512];
    char* homeDir = getenv("HOME");
    char RcName[32] = "/.uni10rc";
    strcpy(RcDir, homeDir);
    strcat(RcDir, RcName);

    FILE* frc = fopen(RcDir, "r");
    if(frc != NULL){
      char line[256];
      std::string Key;
      std::map<std::string, std::vector<std::string> > Key_vals;
      while(fgets(line, 256, frc)){
	line[strlen(line)-1] = ' ';
	char* pch;
	pch = strtok(line, " ,:");
	Key = std::string(pch);
	while(pch != NULL){
	  pch = strtok(NULL, " ,:");
	  if(pch != NULL){
	    //std::cout << "Val: " << std::string(pch) << std::endl;
	    Key_vals[Key].push_back(std::string(pch));
	  }
	}
      }
      if(Key_vals["DEV"].size() != 0)
	ONGPU = true;
      assert(Key_vals["GMODE"].size() == 1);
      if(Key_vals["GMODE"][0] == "Fast")
	GMODE =2;
      fclose(frc);
      //check memory usage of gpu is not ready.
    }
    else{
      GMODE =  GMODE_;
      if(GMODE == 0)
	ONGPU = false;
      else
	ONGPU = true;
      //check memory usage of gpu is not ready.
    }

    if(GMODE != 0){
      size_t free_db, total_db;
      magma_init();
      checkCudaError(cudaMemGetInfo(&free_db, &total_db)); 
      GPU_FREE_MEM = free_db;
      if(printUsage)
	fprintf(stderr, "gpu memory info: total memory: %.ld bytes, free memory: %.ld bytes, usage: %.ld bytes\n", total_db, free_db, total_db-free_db);
    }
    //std::cout << "false" << std::endl;
    //exit(0);
  }

  void guni10_destory(){

    magma_finalize();

  }

  void* elemAlloc(size_t memsize, bool& ongpu){
    void* ptr = NULL;
    if(GPU_FREE_MEM - memsize > 0){
      checkCudaError(cudaMallocManaged(&ptr, memsize));
      GPU_FREE_MEM -= memsize;
      ongpu = true;
    }else{
      ptr = malloc(memsize);
      assert(ptr != NULL);
      MEM_USAGE += memsize;
      ongpu = false;
    }
    //printf("ongpu = %d, GPU_MEM_USAGE = %u, allocate %u\n", ongpu, GPU_MEM_USAGE, memsize);
    return ptr;
  }

  void* elemAllocForce(size_t memsize, bool ongpu){

    void* ptr = NULL;

    if(ongpu){
      checkCudaError(cudaMallocManaged(&ptr, memsize));
      GPU_FREE_MEM -= memsize;
    }
    else{
      ptr = malloc(memsize);
      MEM_USAGE += memsize;
      assert(ptr != NULL);
    }
    return ptr;

  }

  //==================     Magma    =====================/
  
  void* elemAlloc(size_t m, size_t n, size_t typesize, bool& ongpu, bool diag){

    void* ptr = NULL;
    size_t memsize = 0; 

    if(GPU_FREE_MEM - memsize > 0){

      memsize = diag ? magma_roundup(std::min(m, n), blocksize) * typesize : m * magma_roundup(n, blocksize) * typesize;
      magma_malloc((void**)&ptr, memsize);
      GPU_FREE_MEM -= memsize;
      ongpu = true;

    }
    else{

      memsize = diag ? std::min(m, n) * typesize : m * n * typesize;
      ptr = malloc(memsize);
      MEM_USAGE += memsize;
      ongpu = false;
      assert(ptr != NULL);

    }

    return ptr;

  }

  void* elemAllocForce(size_t m, size_t n, size_t typesize, bool ongpu, bool diag){

    void* ptr = NULL;
    size_t memsize = 0;
    //printf("m: %ld, n: %ld.\n\n", m, n);
    if(ongpu){

      //magma_dmalloc(&ptr, m*magma_roundup(n, blocksize));
      memsize = diag ? magma_roundup(std::min(m, n), blocksize) * typesize : m * magma_roundup(n, blocksize) * typesize;
      magma_malloc((void**)&ptr, memsize);
      GPU_FREE_MEM -= memsize;

    }
    else{

      memsize = diag ? std::min(m, n) * typesize : m * n * typesize;
      ptr = (double*)malloc(memsize);
      MEM_USAGE += memsize;
      assert(ptr != NULL);

    }
    return ptr;

  }

  void* elemCopy(double* des, const double* src, size_t m, size_t n, bool des_ongpu, bool src_ongpu, bool diag){

    if(diag){

      n = std::min(m, n); 
      m = 1;

    }
    
    if(des_ongpu){

      if(src_ongpu){

	size_t ldd = magma_roundup(n, blocksize);
	magma_dcopymatrix(n, m, src, ldd, des, ldd);

      }
      else{

	magma_dsetmatrix(n, m, src, n, (magmaDouble_ptr)des, magma_roundup(n, blocksize));

      }

    }else{

      if(src_ongpu){

	magma_dgetmatrix(n, m, src, magma_roundup(n, blocksize), des, n);
      }
      else{

	memcpy(des, src, m * n * sizeof(double));

      }

    }

    return des;

  }

  void* elemCopy(std::complex<double>* des, const std::complex<double>* src, size_t m, size_t n, bool des_ongpu, bool src_ongpu, bool diag){
    
    if(diag){
      n = std::min(m, n); 
      m = 1;
    }

    if(des_ongpu){

      if(src_ongpu){

	size_t ldd = magma_roundup(n, blocksize);
	magma_zcopymatrix(n, m, (magmaDoubleComplex*)src, ldd, (magmaDoubleComplex*)des, ldd);

      }
      else{

	magma_zsetmatrix(n, m, (magmaDoubleComplex*)src, n, (magmaDoubleComplex*)des, magma_roundup(n, blocksize));

      }
    }else{

      if(src_ongpu){

	magma_zgetmatrix(n, m, (magmaDoubleComplex*)src, magma_roundup(n, blocksize), (magmaDoubleComplex*)des, n);

      }
      else{

        memcpy(des, src, m * n * sizeof(std::complex<double>));

      }

    }

    return des;

  }

  void elemRand(double* elem, size_t m, size_t n, bool ongpu, bool diag){

    magma_int_t elemNum = 0;

    if(diag){

      elemNum = std::min(m, n);
      n = std::min(m, n);
      m = 1;

    }
    else{

      elemNum = n * m;

    }

    double* h_elem = (double*)malloc(elemNum*sizeof(double));

    magma_int_t ione = 1;

    magma_int_t ISEED[4] = {0, 0, 0, 1};

    lapackf77_dlarnv(&ione, ISEED, &elemNum, h_elem);

    if(ongpu)
      magma_dsetmatrix(n, m, h_elem, n, elem, magma_roundup(n, blocksize));
    else
      memcpy(elem, h_elem, elemNum*sizeof(double));

  }

  void elemBzero(double* ptr, size_t m, size_t n, bool ongpu, bool diag){
    
    size_t memsize = 0;

    if(ongpu){

      memsize = diag ? magma_roundup( std::min(m, n),  blocksize ) * sizeof(double) : m * magma_roundup(n, blocksize) * sizeof(double);

      checkCudaError(cudaMemset(ptr, 0, memsize));

    }
    else{

      memsize = diag ? std::min(m, n) * sizeof(double) : m * n * sizeof(double);

      memset(ptr, 0, memsize);

    }

  }

  void elemBzero(std::complex<double>* ptr, size_t m, size_t n, bool ongpu, bool diag){
    
    size_t memsize = 0;

    if(ongpu){

      memsize = diag ? magma_roundup(std::min(m, n), blocksize) * sizeof(std::complex<double>) : m * magma_roundup(n, blocksize) * sizeof(std::complex<double>);

      checkCudaError(cudaMemset(ptr, 0, memsize));

    }
    else{

      memsize = diag ? std::min(m, n) * sizeof(std::complex<double>) : m * n * sizeof(std::complex<double>);

      memset(ptr, 0, memsize);

    }

  }

//#ifdef CUDA_SUPPORT
  void magma_to_cuda(size_t m, size_t n, double* mag_ptr, size_t ldda, double* cu_ptr){
      
      for(size_t i = 0; i < m; i++)
	checkCudaError(cudaMemcpy(cu_ptr+(i*n), mag_ptr+(i*ldda), n * sizeof(double), cudaMemcpyDeviceToDevice));

  }

  void cuda_to_magma(size_t m, size_t n, double* cu_ptr, double* mag_ptr, size_t ldda){

      for(size_t i = 0; i < m; i++)
	checkCudaError(cudaMemcpy(mag_ptr+(i*ldda), cu_ptr+(i*n), n * sizeof(double), cudaMemcpyDeviceToDevice));

  }
//#endif
  //=====================================================/
  
  void* elemCopy(void* des, const void* src, size_t memsize,bool des_ongpu, bool src_ongpu){

    if((des_ongpu)){

      if(src_ongpu){
	checkCudaError(cudaMemcpy(des, src, memsize, cudaMemcpyDeviceToDevice));
      }
      else{
	checkCudaError(cudaMemcpy(des, src, memsize, cudaMemcpyHostToDevice));
      }
      //printf("mvGPU\n");
    }else{

      if(src_ongpu){
	checkCudaError(cudaMemcpy(des, src, memsize, cudaMemcpyDeviceToHost));
      }
      else{
	memcpy(des, src, memsize);
      }
      //printf("mvCPU\n");
    }
    return des;

  }


  void elemFree(void* ptr, size_t memsize, bool ongpu){

    assert(ptr != NULL);

    if(ongpu){

      magma_free(ptr);

      GPU_FREE_MEM += memsize;

    }else{

      free(ptr);

      MEM_USAGE -= memsize;
    }

    ptr = NULL;

  }

  void elemBzero(void* ptr, size_t memsize, bool ongpu){
    
    if(ongpu){

      checkCudaError(cudaMemset(ptr, 0, memsize));

    }
    else{

      memset(ptr, 0, memsize);

    }

  }

  void* mvGPU(void* elem, size_t memsize, bool& ongpu){ 
    if(!ongpu && GMODE != 0){
      void* newElem;
      if( GMODE == 2 ){
	ongpu = true;
	newElem = elemAllocForce(memsize, ongpu);
      }
      else if( GMODE == 1 ) //&& GPU_MEM_USAGE + memsize <= GPU_MEM_MAX)
	newElem = elemAlloc(memsize, ongpu);
      elemCopy(newElem, elem, memsize, ongpu, false);
      elemFree(elem, memsize, false);
      elem = newElem;
    }
    return elem;
  }

  void* mvCPU(void* elem, size_t memsize, bool& ongpu){ //Force to CPU
    if(ongpu){
      double *newElem = (double*)malloc(memsize);
      elemCopy(newElem, elem, memsize, false, true);
      elemFree(elem, memsize, true);
      MEM_USAGE += memsize;
      ongpu = false;
      elem = newElem;
    }
    return elem;
  }

  void syncMem(void** elemA, void** elemB, size_t memsizeA, size_t memsizeB, bool& ongpuA, bool& ongpuB){
    if((!ongpuA) || (!ongpuB)){
      size_t memsize = 0;
      if(!ongpuA)
	memsize += memsizeA;
      if(!ongpuB)
	memsize += memsizeB;
      if(GPU_FREE_MEM - memsize > 0){
	if(!ongpuA)
	  *elemA = mvGPU(*elemA, memsizeA, ongpuA);
	if(!ongpuB)
	  *elemB = mvGPU(*elemB, memsizeB, ongpuB);
      }
      else{
	if(ongpuA)
	  *elemA = mvCPU(*elemA, memsizeA, ongpuA);
	if(ongpuB)
	  *elemB = mvCPU(*elemB, memsizeB, ongpuB);
      }
    }
  }
  void shrinkWithoutFree(size_t memsize, bool ongpu){
    if(ongpu)
      GPU_FREE_MEM += memsize;
    else
      MEM_USAGE -= memsize;
  }

  double getElemAt(size_t idx, double* elem, bool ongpu){
    if(ongpu){
      double val;
      assert(cudaMemcpy(&val, &(elem[idx]), sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);
      return val;
    }
    else
      return elem[idx];
  }

  void setElemAt(size_t idx, double val, double* elem, bool ongpu){
    if(ongpu){
      assert(cudaMemcpy(&(elem[idx]), &val, sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
    }
    else
      elem[idx] = val;
  }

  double  elemMax(double* elem, size_t elemNum, bool ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  double  elemAbsMax(double* elem, size_t elemNum, bool ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  std::complex<double> getElemAt(size_t idx, std::complex<double>* elem, bool ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void setElemAt(size_t idx, std::complex<double> val, std::complex<double>* elem, bool ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void elemRand(std::complex<double>* elem, size_t N, bool ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void elemCast(std::complex<double>* des, double* src, size_t N, bool des_ongpu, bool src_ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void elemCast(double *des, std::complex<double> *src, size_t N, bool des_ongpu, bool src_ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void setDiag(std::complex<double>* elem, std::complex<double>* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void getDiag(std::complex<double>* elem, std::complex<double>* diag_elem, size_t M, size_t N, size_t diag_N, bool ongpu, bool diag_ongpu){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

  void reshapeElem(std::complex<double>* oldElem, int bondNum, size_t elemNum, size_t* offset, std::complex<double>* newElem){

    std::ostringstream err;
    err<<"GPU version is not ready !!!!";
    throw std::runtime_error(exception_msg(err.str()));

  }

};	/* namespace uni10 */
