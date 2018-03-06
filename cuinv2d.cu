#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#ifdef _WIN32
    #include <direct.h>
#elif defined __linux__
    #include <sys/stat.h>
    #include <sys/types.h>
    #define _mkdir(dir) mkdir(dir, 0777)
#endif

#define devij int i = blockIdx.x, j = threadIdx.x
#define devij2 int i0 = blockIdx.x, i = blockIdx.x+blockDim.x*blockIdx.y, j = threadIdx.x;

const float pi = 3.1415927;
__constant__ float d_pi = 3.1415927;

cublasHandle_t cublas_handle;
cusolverDnHandle_t solver_handle;
cufftHandle cufft_handle;

namespace dat{
    int nx;
    int nx2;
    int nz;
    int nt;
    float dt;
    float Lx;
    float Lz;
    float **X;
    float **Z;

    dim3 nxb;
    dim3 nxb2;
    dim3 nzt;

    int sfe;
    int nsfe;
    int order;
    int wave_propagation_sh;
    int wave_propagation_psv;
    int simulation_mode;
    int display_mode;

    int absorb_left;
    int absorb_right;
    int absorb_top;
    int absorb_bottom;
    float absorb_width;

    int *isrc;
    int nsrc;
    int nrec;
    int ntask;
    int obs_type;
    int obs_su;
    int src_file;
    int rec_file;
    int src_align;
    int rec_align;
    int misfit_type;
    int parametrisation;

    char *parfile;
    char *argstr;
    char *obs_su_path;
    char *src_file_path;
    char *rec_file_path;
    char *model_init;
    char *model_true;
    char *output_path;

    float model_p;
    float model_s;
    float model_r;

    int *src_type;  // host (ricker = 1)
    float *src_angle;  // host
    float *src_f0;  // host
    float *src_t0;  // host
    float *src_factor; // host
    float *src_x;
    float *src_z;

    float *rec_x;
    float *rec_z;

    int *src_x_id;
    int *src_z_id;
    int *rec_x_id;
    int *rec_z_id;

    float **stf_x;
    float **stf_y;
    float **stf_z;
    float **adstf_x;
    float **adstf_y;
    float **adstf_z;

    float **lambda;
    float **mu;
    float **rho;
    float **absbound;

    float **ux;
    float **uy;
    float **uz;
    float **vx;
    float **vy;
    float **vz;

    float **sxx;
    float **sxy;
    float **sxz;
    float **szy;
    float **szz;

    float **dsx;
    float **dsy;
    float **dsz;
    float **dvxdx;
    float **dvxdz;
    float **dvydx;
    float **dvydz;
    float **dvzdx;
    float **dvzdz;

    float **dvxdx_fw;
    float **dvxdz_fw;
    float **dvydx_fw;
    float **dvydz_fw;
    float **dvzdx_fw;
    float **dvzdz_fw;

    float **K_lambda;
    float **K_mu;
    float **K_rho;

    float **out_x;
    float **out_y;
    float **out_z;

    float ***u_obs_x;
    float ***u_obs_y;
    float ***u_obs_z;

    float ***ux_forward;  // host
    float ***uy_forward;  // host
    float ***uz_forward;  // host
    float ***vx_forward;  // host
    float ***vy_forward;  // host
    float ***vz_forward;  // host

    int filter_kernel;
    float unsharp_mask;
    float misfit_ref;
    float **gsum;
    float **gtemp;
    float *tw;

    int optimize;
    int inv_parameter;
    int inv_iteration;
    int ls_stepcountmax;
    int ls_count;

    float ls_thresh;
    float ls_steplenmax;
    float ls_stepleninit;

    float *func_vals;  // host
    float *step_lens;  // host
    float *ls_gtp;  // host
    float *ls_gtg;  // host
    float **opti_history; // host

    int inv_count;
    int inv_maxiter;
    int lbfgs_mem;

    float ***lbfgs_S;  // host
    float ***lbfgs_Y;  // host
    int lbfgs_used;

    FILE *logfile;
    int neval;
}
namespace mat{
    __global__ void _setValue(float *mat, const float init){
        int i = blockIdx.x;
        mat[i] = init;
    }
    __global__ void _setValue(double *mat, const double init){
        int i = blockIdx.x;
        mat[i] = init;
    }
    __global__ void _setValue(float **mat, const float init){
        devij;
        mat[i][j] = init;
    }
    __global__ void _setValue(float ***mat, const float init, const int p){
        devij;
        mat[p][i][j] = init;
    }
    __global__ void _setPointerValue(float **mat, float *data, const int n){
        int i = blockIdx.x;
        mat[i] = data + n * i;
    }
    __global__ void _setPointerValue(float ***mat, float **data, const int i){
        mat[i] = data;
    }
    __global__ void _setIndexValue(float *a, float *b, int index){
        a[0] = b[index];
    }
    __global__ void _copy(float *mat, float *init){
        int i = blockIdx.x;
        mat[i] = init[i];
    }
    __global__ void _copy(float **mat, float **init){
        devij;
        mat[i][j] = init[i][j];
    }
    __global__ void _copy(float **mat, float **init, float k){
        devij;
        mat[i][j] = init[i][j] * k;
    }
    __global__ void _calc(float **c, float ka, float **a, float kb, float **b){
        devij;
        c[i][j] = ka * a[i][j] + kb * b[i][j];
    }
    __global__ void _calc(float *c, float ka, float *a, float kb, float *b){
        int i = blockIdx.x;
        c[i] = ka * a[i] + kb * b[i];
    }
    __global__ void _calc(float *c, float *a, float *b){
        int i = blockIdx.x;
        c[i] = a[i] * b[i];
    }

    float *init(float *mat, const float init, const int m){
        mat::_setValue<<<m, 1>>>(mat, init);
        return mat;
    }
    double *init(double *mat, const double init, const int m){
        mat::_setValue<<<m, 1>>>(mat, init);
        return mat;
    }
    float **init(float **mat, const float init, const int m, const int n){
        mat::_setValue<<<m, n>>>(mat, init);
        return mat;
    }
    float ***init(float ***mat, const float init, const int p, const int m, const int n){
        for(int i = 0; i < p; i++){
            mat::_setValue<<<m, n>>>(mat, init, i);
        }
        return mat;
    }
    float *initHost(float *mat, const float init, const int m){
        for(int i = 0; i < m; i++){
            mat[i] = init;
        }
        return mat;
    }
    float **initHost(float **mat, const float init, const int m, const int n){
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                mat[i][j] = init;
            }
        }
        return mat;
    }
    float ***initHost(float ***mat, const float init, const int p, const int m, const int n){
        for(int k = 0; k < p; k++){
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    mat[k][i][j] = init;
                }
            }
        }
        return mat;
    }

    float *create(const int m) {
    	float *data;
    	cudaMalloc((void **)&data, m * sizeof(float));
    	return data;
    }
    float **create(const int m, const int n){
    	float *data = mat::create(m * n);
        float **mat;
        cudaMalloc((void **)&mat, m * sizeof(float *));
        mat::_setPointerValue<<<m, 1>>>(mat, data, n);
    	return mat;
    }
    float ***create(const int p, const int m, const int n){
        float ***mat;
        cudaMalloc((void **)&mat, p * sizeof(float **));
        for(int i = 0; i < p; i++){
            mat::_setPointerValue<<<1,1>>>(mat, mat::create(m, n), i);
        }
        return mat;
    }
    float *createHost(const int m) {
    	return (float *)malloc(m * sizeof(float));
    }
    float **createHost(const int m, const int n){
        float *data = mat::createHost(m * n);
    	float **mat = (float **)malloc(m * sizeof(float *));
    	for(int i  =0; i < m; i++){
    		mat[i] = data + n * i;
    	}
    	return mat;
    }
    float ***createHost(const int p, const int m, const int n){
        float ***mat = (float ***)malloc(p * sizeof(float **));
        for(int i = 0; i < p; i++){
            mat[i] = mat::createHost(m, n);
        }
        return mat;
    }
    int *createInt(const int m){
        int *a;
    	cudaMalloc((void**)&a, m * sizeof(int));
    	return a;
    }
    int *createIntHost(const int m) {
    	return (int *)malloc(m * sizeof(int));
    }
    short int *createShortInt(const int m){
        short int *a;
    	cudaMalloc((void**)&a, m * sizeof(short int));
    	return a;
    }
    short int *createShortIntHost(const int m){
        return (short int *)malloc(m * sizeof(short int));
    }
    double *createDouble(const int m){
        double *a;
    	cudaMalloc((void**)&a, m * sizeof(double));
    	return a;
    }
    double *createDoubleHost(const int m) {
    	return (double *)malloc(m * sizeof(double));
    }

    float *getDataPointer(float **mat){
        float **p=(float **)malloc(sizeof(float *));
        cudaMemcpy(p, mat , sizeof(float *), cudaMemcpyDeviceToHost);
        return *p;
    }
    void copy(float *mat, float *init, const int m){
        mat::_copy<<<m, 1>>>(mat, init);
    }
    void copy(float **mat, float **init, const int m, const int n){
        mat::_copy<<<m, n>>>(mat, init);
    }
    void copy(float **mat, float **init, float k, const int m, const int n){
        mat::_copy<<<m, n>>>(mat, init, k);
    }
    void copyHostToDevice(float *d_a, const float *a, const int m){
        cudaMemcpy(d_a, a , m * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyHostToDevice(float **pd_a, float *pa, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(*phd_a, pa , m * n * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyHostToDevice(float **pd_a, float **pa, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(*phd_a, *pa , m * n * sizeof(float), cudaMemcpyHostToDevice);
    }
    void copyHostToDevice(float ***pd_a, float ***pa, const int p, const int m, const int n){
        float ***phd_a=(float ***)malloc(p * sizeof(float **));
        cudaMemcpy(phd_a, pd_a, p * sizeof(float **), cudaMemcpyDeviceToHost);
        for(int i = 0; i < p; i++){
            mat::copyHostToDevice(phd_a[i], pa[i], m, n);
        }
    }
    void copyDeviceToHost(float *a, const float *d_a, const int m){
        cudaMemcpy(a, d_a , m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void copyDeviceToHost(float *pa, float **pd_a, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(pa, *phd_a , m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void copyDeviceToHost(float **pa, float **pd_a, const int m, const int n){
        float **phd_a=(float **)malloc(sizeof(float *));
        cudaMemcpy(phd_a, pd_a , sizeof(float *), cudaMemcpyDeviceToHost);
        cudaMemcpy(*pa, *phd_a , m * n * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void copyDeviceToHost(float ***pa, float ***pd_a, const int p, const int m, const int n){
        float ***phd_a=(float ***)malloc(p * sizeof(float **));
        cudaMemcpy(phd_a, pd_a, p * sizeof(float **), cudaMemcpyDeviceToHost);
        for(int i = 0; i < p; i++){
            mat::copyDeviceToHost(pa[i], phd_a[i], m, n);
        }
    }

    void calc(float *c, float *a, float *b, int m){
        mat::_calc<<<m, 1>>>(c, a, b);
    }
    void calc(float *c, float ka, float *a, float kb, float *b, int m){
        mat::_calc<<<m, 1>>>(c, ka, a, kb, b);
    }
    void calc(float **c, float ka, float **a, float kb, float **b, int m, int n){
        mat::_calc<<<m, n>>>(c, ka, a, kb, b);
    }
    float norm(float *a, int n){
        float norm_a = 0;
        cublasSnrm2_v2(cublas_handle, n, a, 1, &norm_a);
        return norm_a;
    }
    float norm(float **a, int m, int n){
        return mat::norm(mat::getDataPointer(a), m * n);
    }
    float amax(float *a, int n){
        int index = 0;
        cublasIsamax_v2(cublas_handle, n, a, 1, &index);
        float *b = mat::create(1);
        mat::_setIndexValue<<<1, 1>>>(b, a, index - 1);
        float *c = mat::createHost(1);
        mat::copyDeviceToHost(c, b, 1);
        return fabs(c[0]);
    }
    float amax(float **a, int m, int n){
        return mat::amax(mat::getDataPointer(a), m * n);
    }
    float dot(float *a, float *b, int n){
        float dot_ab = 0;
        cublasSdot_v2(cublas_handle, n, a, 1, b, 1, &dot_ab);
        return dot_ab;
    }
    float dot(float **a, float **b, int m, int n){
        return mat::dot(mat::getDataPointer(a), mat::getDataPointer(b), m * n);
    }

    void freeHost(float **mat){
        free(*mat);
        free(mat);
    }
    void freeHost(float ***mat){
        free(**mat);
        free(*mat);
        free(mat);
    }
    void freeDevice(float **mat){
        cudaFree(getDataPointer(mat));
        cudaFree(mat);
    }

    void read(float *data, int n, const char *fname){
        FILE *file = fopen(fname, "rb");
        fread(data, sizeof(float), n, file);
        fclose(file);
    }
    void write(float *data, int n, const char *fname){
        FILE *file = fopen(fname, "wb");
        int npt = n * 4;
        fwrite(&npt, sizeof(int), 1, file);
        fwrite(data, sizeof(float), n, file);
        fclose(file);
    }
    void write(float **data, int m, int n, const char *fname){
        FILE *file = fopen(fname, "wb");
        int npt = m * n * 4;
        fwrite(&npt, sizeof(int), 1, file);
        for(int i = 0; i < m; i++){
            fwrite(data[i], sizeof(float), n, file);
        }
        fclose(file);
    }
    void write(float ***data, int p, int m, int n, const char *fname){
        FILE *file = fopen(fname, "wb");
        int npt = p * m * n * 4;
        fwrite(&npt, sizeof(int), 1, file);
        for(int k = 0; k < p; k++){
            for(int i = 0; i < m; i++){
                fwrite(data[k][i], sizeof(float), n, file);
            }
        }
        fclose(file);
    }
    void writeDevice(float *data, int n, const char *fname){
        float *h_data = mat::createHost(n);
        mat::copyDeviceToHost(h_data, data, n);
        mat::write(h_data, n, fname);
        free(h_data);
    }
    void writeDevice(float **data, const int m, int n, const char *fname){
        float **h_data = mat::createHost(m, n);
        mat::copyDeviceToHost(h_data, data, m, n);
        mat::write(h_data, m, n, fname);
        mat::freeHost(h_data);
    }
    void writeDevice(float ***data, const int p, const int m, int n, const char *fname){
        float ***h_data = mat::createHost(p, m, n);
        mat::copyDeviceToHost(h_data, data, p, m, n);
        mat::write(h_data, p, m, n, fname);
        mat::freeHost(h_data);
    }
}

dim3 &nxb = dat::nxb;
dim3 &nxb2 = dat::nxb2;
dim3 &nzt = dat::nzt;

int &sh = dat::wave_propagation_sh;
int &psv = dat::wave_propagation_psv;
int &mode = dat::simulation_mode;
int &ntask = dat::ntask;

int &nx = dat::nx;
int &nx2 = dat::nx2;
int &nz = dat::nz;
int &nt = dat::nt;
int &nsrc = dat::nsrc;
int &nrec = dat::nrec;
float &dt = dat::dt;

__global__ void divSY(float **dsy, float **sxy, float **szy, float **X, float **Z, int nx, int nz){
    devij2;
    if(i0 >= 2 && i0 < nx - 2){
        float dx = X[i0][j] - X[i0-1][j];
        float dx3 = X[i0+1][j] - X[i0-2][j];
        dsy[i][j] = 9*(sxy[i][j] - sxy[i-1][j])/(8*dx) - (sxy[i+1][j] - sxy[i-2][j])/(8*dx3);
    }
    else{
        dsy[i][j] = 0;
    }
    if(j >= 2 && j < nz - 2){
        float dz = Z[i0][j] - Z[i0][j-1];
        float dz3 = Z[i0][j+1] - Z[i0][j-2];
        dsy[i][j] += 9*(szy[i][j] - szy[i][j-1])/(8*dz) - (szy[i][j+1] - szy[i][j-2])/(8*dz3);
    }
}
__global__ void divSXZ(float **dsx, float **dsz, float **sxx, float **szz, float **sxz, float **X, float **Z, int nx, int nz){
    devij2;
    if(i0 >= 2 && i0 < nx - 2){
        float dx = X[i0][j] - X[i0-1][j];
        float dx3 = X[i0+1][j] - X[i0-2][j];
        dsx[i][j] = 9*(sxx[i][j] - sxx[i-1][j])/(8*dx) - (sxx[i+1][j] - sxx[i-2][j])/(8*dx3);
        dsz[i][j] = 9*(sxz[i][j] - sxz[i-1][j])/(8*dx) - (sxz[i+1][j] - sxz[i-2][j])/(8*dx3);
    }
    else{
        dsx[i][j] = 0;
        dsz[i][j] = 0;
    }
    if(j >= 2 && j < nz - 2){
        float dz = Z[i0][j] - Z[i0][j-1];
        float dz3 = Z[i0][j+1] - Z[i0][j-2];
        dsx[i][j] += 9*(sxz[i][j] - sxz[i][j-1])/(8*dz) - (sxz[i][j+1] - sxz[i][j-2])/(8*dz3);
        dsz[i][j] += 9*(szz[i][j] - szz[i][j-1])/(8*dz) - (szz[i][j+1] - szz[i][j-2])/(8*dz3);
    }
}
__global__ void divVY(float **dvydx, float **dvydz, float **vy, float **X, float **Z, int nx, int nz){
    devij2;
    if(i0 >= 1 && i0 < nx - 2){
        float dx = X[i0+1][j] - X[i0][j];
        float dx3 = X[i0+2][j] - X[i0-1][j];
        dvydx[i][j] = 9*(vy[i+1][j] - vy[i][j])/(8*dx) - (vy[i+2][j] - vy[i-1][j])/(8*dx3);
    }
    else{
        dvydx[i][j] = 0;
    }
    if(j >= 1 && j < nz - 2){
        float dz = Z[i0][j+1] - Z[i0][j];
        float dz3 = Z[i0][j+2] - Z[i0][j-1];
        dvydz[i][j] = 9*(vy[i][j+1] - vy[i][j])/(8*dz) - (vy[i][j+2] - vy[i][j-1])/(8*dz3);
    }
    else{
        dvydz[i][j] = 0;
    }
}
__global__ void divVXZ(float **dvxdx, float **dvxdz, float **dvzdx, float **dvzdz, float **vx, float **vz, float **X, float **Z, int nx, int nz){
    devij2;
    if(i0 >= 1 && i0 < nx - 2){
        float dx = X[i0+1][j] - X[i0][j];
        float dx3 = X[i0+2][j] - X[i0-1][j];
        dvxdx[i][j] = 9*(vx[i+1][j]-vx[i][j])/(8*dx)-(vx[i+2][j]-vx[i-1][j])/(8*dx3);
        dvzdx[i][j] = 9*(vz[i+1][j]-vz[i][j])/(8*dx)-(vz[i+2][j]-vz[i-1][j])/(8*dx3);
    }
    else{
        dvxdx[i][j] = 0;
        dvzdx[i][j] = 0;
    }
    if(j >= 1 && j < nz - 2){
        float dz = Z[i0][j+1] - Z[i0][j];
        float dz3 = Z[i0][j+2] - Z[i0][j-1];
        dvxdz[i][j] = 9*(vx[i][j+1]-vx[i][j])/(8*dz)-(vx[i][j+2]-vx[i][j-1])/(8*dz3);
        dvzdz[i][j] = 9*(vz[i][j+1]-vz[i][j])/(8*dz)-(vz[i][j+2]-vz[i][j-1])/(8*dz3);
    }
    else{
        dvxdz[i][j] = 0;
        dvzdz[i][j] = 0;
    }
}

__global__ void addSTF(float **dsx, float **dsy, float **dsz, float **stf_x, float **stf_y, float **stf_z,
    int *src_x_id, int *src_z_id, int isrc, int sh, int psv, int it, int nx, int nt){
    int is = blockIdx.x;
    int xs = src_x_id[is];
    int zs = src_z_id[is];

    int is2 = threadIdx.x;
    if(isrc < 0 || isrc + is2 == is){
        if(sh){
            dsy[xs+is2*nx][zs] += stf_y[is][it+nt*is2];
        }
        if(psv){
            dsx[xs+is2*nx][zs] += stf_x[is][it+nt*is2];
            dsz[xs+is2*nx][zs] += stf_z[is][it+nt*is2];
        }
    }
}
__global__ void saveRec(float **out_x, float **out_y, float **out_z, float **vx, float **vy, float **vz,
    int *rec_x_id, int *rec_z_id, int nx, int nt, int sh, int psv, int it){
    int ir = blockIdx.x;
    int xr = rec_x_id[ir] + threadIdx.x * nx;
    int zr = rec_z_id[ir];

    it += threadIdx.x * nt;
    if(sh){
        out_y[ir][it] = vy[xr][zr];
    }
    if(psv){
        out_x[ir][it] = vx[xr][zr];
        out_z[ir][it] = vz[xr][zr];
    }
}
__global__ void saveRec(float ***out_x, float ***out_y, float ***out_z, float **vx, float **vy, float **vz,
    int *rec_x_id, int *rec_z_id, int isrc, int nx, int sh, int psv, int it){
    int ir = blockIdx.x;
    int xr = rec_x_id[ir];
    int zr = rec_z_id[ir];

    int is2 = threadIdx.x;
    if(sh){
        out_y[isrc+is2][ir][it] = vy[xr+is2*nx][zr];
    }
    if(psv){
        out_x[isrc+is2][ir][it] = vx[xr+is2*nx][zr];
        out_z[isrc+is2][ir][it] = vz[xr+is2*nx][zr];
    }
}
__global__ void updateV(float **v, float **ds, float **rho, float **absbound, float dt){
    devij2;
    v[i][j] = absbound[i0][j] * (v[i][j] + dt * ds[i][j] / rho[i0][j]);
}
__global__ void updateSY(float **sxy, float **szy, float **dvydx, float **dvydz, float **mu, float dt){
    devij2;
    sxy[i][j] += dt * mu[i0][j] * dvydx[i][j];
    szy[i][j] += dt * mu[i0][j] * dvydz[i][j];
}
__global__ void updateSXZ(float **sxx, float **szz, float **sxz, float **dvxdx, float **dvxdz, float **dvzdx, float **dvzdz,
    float **lambda, float **mu, float dt){
    devij2;
    sxx[i][j] += dt * ((lambda[i0][j] + 2 * mu[i0][j]) * dvxdx[i][j] + lambda[i0][j] * dvzdz[i][j]);
    szz[i][j] += dt * ((lambda[i0][j] + 2 * mu[i0][j]) * dvzdz[i][j] + lambda[i0][j] * dvxdx[i][j]);
    sxz[i][j] += dt * (mu[i0][j] * (dvxdz[i][j] + dvzdx[i][j]));
}
__global__ void updateU(float **u, float **v, float dt){
    int i = blockIdx.x+blockDim.x*blockIdx.y, j = threadIdx.x;
    u[i][j] += v[i][j] * dt;
}
__global__ void interactionRhoY(float **K_rho, float **vy, float **vy_fw, float tsfe){
    devij2;
    K_rho[i0][j] -= vy_fw[i][j] * vy[i][j] * tsfe;
}
__global__ void interactionRhoXZ(float **K_rho, float **vx, float **vx_fw, float **vz, float **vz_fw, float tsfe){
    devij2;
    K_rho[i0][j] -= (vx_fw[i][j] * vx[i][j] + vz_fw[i][j] * vz[i][j]) * tsfe;
}
__global__ void interactionMuY(float **K_mu, float **dvydx, float **dvydx_fw, float **dvydz, float **dvydz_fw, float tsfe){
    devij2;
    K_mu[i0][j] -= (dvydx[i][j] * dvydx_fw[i][j] + dvydz[i][j] * dvydz_fw[i][j]) * tsfe;
}
__global__ void interactionMuXZ(float **K_mu, float **dvxdx, float **dvxdx_fw, float **dvxdz, float **dvxdz_fw,
    float **dvzdx, float **dvzdx_fw, float **dvzdz, float **dvzdz_fw, float tsfe){
    devij2;
    K_mu[i0][j] -= (2 * dvxdx[i][j] * dvxdx_fw[i][j] + 2 * dvzdz[i][j] * dvzdz_fw[i][j] +
        (dvxdz[i][j] + dvzdx[i][j]) * (dvzdx_fw[i][j] + dvxdz_fw[i][j])) * tsfe;
}
__global__ void interactionLambdaXZ(float **K_lambda, float **dvxdx, float **dvxdx_fw, float **dvzdz, float **dvzdz_fw, float tsfe){
    devij2;
    K_lambda[i0][j] -= ((dvxdx[i][j] + dvzdz[i][j]) * (dvxdx_fw[i][j] + dvzdz_fw[i][j])) * tsfe;
}

__device__ float gaussian(int x, int sigma){
    float xf = (float)x;
    float sigmaf = (float)sigma;
    return (1 / (sqrtf(2 * d_pi) * sigmaf)) * expf(-xf * xf / (2 * sigmaf * sigmaf));
}
__global__ void initialiseGaussian(float **model, int nx, int nz, int sigma){
    devij;
    float sumx = 0;
    for(int n = 0; n < nx; n++){
        sumx += gaussian(i - n, sigma);
    }
    float sumz = 0;
    for(int n = 0; n < nz; n++){
        sumz += gaussian(j - n, sigma);
    }
    model[i][j] = sumx * sumz;
}
__global__ void computeIndices(int *coord_n_id, float *coord_n, float Ln, float n){
    int i = blockIdx.x;
    coord_n_id[i] = (int)(coord_n[i] / Ln * (n - 1) + 0.5);
}
__global__ void initialiseAbsorbingBoundaries(float **absbound, float width,
    int absorb_left, int absorb_right, int absorb_bottom, int absorb_top,
    float Lx, float Lz, float **X, float **Z){
    devij;
    absbound[i][j] = 1;

    if(absorb_left){
        if(X[i][j] < width){
            absbound[i][j] *= exp(-pow((X[i][j] - width) / (2 * width), 2));
        }
    }
    if(absorb_right){
        if(X[i][j] > Lx - width){
            absbound[i][j] *= exp(-pow((X[i][j] - (Lx - width)) / (2 * width), 2));
        }
    }
    if(absorb_bottom){
        if(Z[i][j] < width){
            absbound[i][j] *= exp(-pow((Z[i][j] - width) / (2 * width), 2));
        }
    }
    if(absorb_top){
        if(Z[i][j] > Lz - width){
            absbound[i][j] *= exp(-pow((Z[i][j] - (Lz - width)) / (2 * width), 2));
        }
    }
}
__global__ void prepareEnvelopeSTF(float **adstf, float *etmp, float *syn, float *ersd, int nt, int irec, int jnt){
    int it = blockIdx.x;
    adstf[irec][nt-it-1 + jnt] = etmp[it] * syn[it] - ersd[it];
}
__global__ void filterKernelX(float **model, float **gtemp, int nx, int sigma){
    devij;
    float sumx = 0;
    for(int n = 0; n < nx; n++){
        sumx += gaussian(i - n, sigma) * model[n][j];
    }
    gtemp[i][j] = sumx;
}
__global__ void filterKernelZ(float **model, float **gtemp, float **gsum, int nz, int sigma){
    devij;
    float sumz = 0;
    for(int n = 0; n < nz; n++){
        sumz += gaussian(j - n, sigma) * gtemp[i][n];
    }
    model[i][j] = sumz / gsum[i][j];
}
__global__ void getTaperWeights(float *tw, float dt, int nt){
    int it = blockIdx.x;

    float t_end = (nt - 1) * dt;
    float taper_width = t_end / 10;
    float t_min = taper_width;
    float t_max = t_end - taper_width;

    float t = it * dt;
    if(t <= t_min){
        tw[it] = 0.5 + 0.5 * cosf(d_pi * (t_min - t) / (taper_width));
    }
    else if(t >= t_max){
        tw[it] = 0.5 + 0.5 * cosf(d_pi * (t_max - t) / (taper_width));
    }
    else{
        tw[it] = 1;
    }
}
__global__ void calculateWaveformMisfit(float **adstf, float *misfit, float **u_syn, float ***u_obs, float *tw, int isrc, int irec, int j, int nt){
    int it = blockIdx.x;
    int jnt = j * nt;
    misfit[it] = (u_syn[irec][it+jnt] - u_obs[isrc+j][irec][it]) * tw[it];
    adstf[irec][nt-it-1 + jnt] = misfit[it] * tw[it] * 2;
}
__global__ void envelopetmp(float *etmp, float *esyn, float *eobs, float max){
    int it = blockIdx.x;
    etmp[it] = (esyn[it] - eobs[it])/(esyn[it] + max);
}
__global__ void copyWaveform(float *misfit, float ***u_obs, int isrc, int irec){
    int it = blockIdx.x;
    misfit[it] = u_obs[isrc][irec][it];
}
__global__ void copyWaveform(float *misfit, float **out, int irec, int jnt){
    int it = blockIdx.x;
    misfit[it] = out[irec][it+jnt];
}
__global__ void initialiseGrids(float **X, float **Z, float Lx, int nx, float Lz, int nz){
    devij;
    X[i][j] = Lx / (nx - 1) * i;
    Z[i][j] = Lz / (nz - 1) * j;
}
__global__ void mesh2grid(float *xbuffer, float *zbuffer, float *rbuffer, float *pbuffer, float *sbuffer,
    float **lambda, float **mu, float **rho, float dx, float dz, float dmax, int npt){
    devij;
    float ix = i * dx;
    float iz = j * dz;
    float dmin = dmax;
    for(int k = 0; k < npt; k++){
        float dx = ix - xbuffer[k];
        float dz = iz - zbuffer[k];
        float d = dx * dx + dz * dz;
        if(d < dmin){
            dmin = d;
            rho[i][j] = rbuffer[k];
            mu[i][j] = sbuffer[k];
            lambda[i][j] = pbuffer[k];
        }
    }
}
__global__ void changeParametrisation(float **lambda, float **mu, float **rho, int psv){
    devij;
    if(psv){
        lambda[i][j] = rho[i][j] * (lambda[i][j] * lambda[i][j] - 2 * mu[i][j] * mu[i][j]);
    }
    else{
        lambda[i][j] = 0;
    }
    mu[i][j] = rho[i][j] * mu[i][j] * mu[i][j];
}
__global__ void changeParametrisation(float *vp, float *vs, float *rho, int nz, int psv){
    devij;
    int ij = i * nz + j;
    if(psv){
        vp[ij] = sqrt((vp[ij] + 2*vs[ij]) / rho[ij]);
    }
    else{
        vp[ij] = 0;
    }
    vs[ij] = sqrt(vs[ij] / rho[ij]);
}
__global__ void updateModel(float **m, float **p, float alpha){
    devij;
    m[i][j] += alpha * p[i][j];
}
__global__ void reduceSystem(const double * __restrict d_in1, double * __restrict d_out1, const double * __restrict d_in2, double * __restrict d_out2, const int M, const int N) {
    const int i = blockIdx.x;
    const int j = threadIdx.x;

    if ((i < N) && (j < N)){
        d_out1[j * N + i] = d_in1[j * M + i];
        d_out2[j * N + i] = d_in2[j * M + i];
    }
}
__global__ void generateChecker(float **p, float dp, float margin, float lx, float lz, float **X, float **Z){
    devij;
    float x = X[i][j];
    float z = Z[i][j];
    float marginx = lx * margin;
    float marginz = lz * margin;
    int idx = (int)((x - marginx*2) / (lx + marginx));
    int idz = (int)((z - marginz*2) / (lz + marginz));
    float rx = x - marginx*2 - idx * (lx + marginx);
    float rz = z - marginz*2 - idz * (lz + marginz);

    if(rx > 0 && rx < lx && rz > 0 && rz < lz){
        if(idx % 2 == idz % 2){
            p[i][j] *= (1 + dp) * (1 + dp);
        }
        else{
            p[i][j] *= (1 - dp) * (1 - dp);
        }
    }
}
__global__ void generateLayer(float **p, float from, float to, float value, float **Z){
    devij;
    float z = Z[i][j];
    if(z >=from && z <= to){
        p[i][j] *= (1+value) * (1+value);
    }
}
__global__ void generateRandomLayer(float **p, float from, float to, float value, float *layer1, float *layer2, float **Z){
    devij;
    float z = Z[i][j];
    if(z >=from+layer1[i] && z <= to+layer2[i]){
        p[i][j] *= (1+value) * (1+value);
    }
}
__global__ void hilbert(cufftComplex *h, int n){
    int i = blockIdx.x;
    if(i > 0){
        if(n % 2 == 0){
            if(i < n / 2 + 1){
                h[i].x *= 2;
                h[i].y *= 2;
            }
            else if(i > n / 2 + 1){
                h[i].x = 0;
                h[i].y = 0;
            }
        }
        else{
            if(i < (n+1) / 2){
                h[i].x *= 2;
                h[i].y *= 2;
            }
            else{
                h[i].x = 0;
                h[i].y = 0;
            }
        }
    }
}
__global__ void copyR2C(cufftComplex *a,float *b){
    int i=blockIdx.x;
    a[i].x=b[i];
    a[i].y=0;
}
__global__ void copyC2Real(float *a, cufftComplex *b, int n){
    int i = blockIdx.x;
    a[i] = b[i].x / n;
}
__global__ void copyC2Imag(float *a, cufftComplex *b, int n){
    int i = blockIdx.x;
    a[i] = b[i].y / n;
}
__global__ void copyC2Abs(float *a, cufftComplex *b, int n){
    int i = blockIdx.x;
    a[i] = sqrt(b[i].x*b[i].x + b[i].y*b[i].y) / n;
}

static void createDirectory(const char *dir){
    char path[80];
    int len = strlen(dir);
    for(int i = 0; i < len; i++){
        if(dat::output_path[i] == '/'){
            path[i] = '\0';
            _mkdir(path);
        }
        path[i] = dat::output_path[i];
    }
    _mkdir(dir);
}
static char *copyString(const char *str){
    char *out = (char *)malloc((strlen(str)+1) * sizeof(char));
    strcpy(out, str);
    return out;
}
static int getTaskIndex(int isrc){
    int index = isrc + ntask - 1;
    if(index >= nsrc){
        return nsrc - 1;
    }
    else{
        return index;
    }
}
static float calculateAngle(float **p, float **g, float k, int nx, int nz){
    float xx = mat::dot(p, p, nx, nz);
    float yy = mat::dot(g, g, nx, nz);
    float xy = k * mat::dot(p, g, nx, nz);
    return acos(xy / sqrt(xx * yy));
}
static void hilbert(float *x, cufftComplex *data){
    copyR2C<<<nt, 1>>>(data, x);
    cufftExecC2C(cufft_handle, data, data, CUFFT_FORWARD);
    hilbert<<<nt,1>>>(data, nt);
    cufftExecC2C(cufft_handle, data, data, CUFFT_INVERSE);
}
static void solveQR(double *h_A, double *h_B, double *XC, const int Nrows, const int Ncols){
    int work_size = 0;
    int *devInfo = mat::createInt(1);

    double *d_A = mat::createDouble(Nrows * Ncols);
    cudaMemcpy(d_A, h_A, Nrows * Ncols * sizeof(double), cudaMemcpyHostToDevice);

    double *d_TAU = mat::createDouble(min(Nrows, Ncols));
    cusolverDnDgeqrf_bufferSize(solver_handle, Nrows, Ncols, d_A, Nrows, &work_size);
    double *work = mat::createDouble(work_size);

    cusolverDnDgeqrf(solver_handle, Nrows, Ncols, d_A, Nrows, d_TAU, work, work_size, devInfo);

    double *d_Q = mat::createDouble(Nrows * Nrows);
    cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_Q, Nrows, work, work_size, devInfo);

    double *d_C = mat::createDouble(Nrows * Nrows);
    mat::init(d_C, 0, Nrows * Nrows);
    cudaMemcpy(d_C, h_B, Nrows * sizeof(double), cudaMemcpyHostToDevice);

    cusolverDnDormqr(solver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, Nrows, Ncols, min(Nrows, Ncols), d_A, Nrows, d_TAU, d_C, Nrows, work, work_size, devInfo);

    double *d_R = mat::createDouble(Ncols * Ncols);
    double *d_B = mat::createDouble(Ncols * Ncols);
    reduceSystem<<<Ncols, Ncols>>>(d_A, d_R, d_C, d_B, Nrows, Ncols);

    const double alpha = 1.;
    cublasDtrsm(cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, Ncols, Ncols,
        &alpha, d_R, Ncols, d_B, Ncols);
    cudaMemcpy(XC, d_B, Ncols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Q);
    cudaFree(d_R);
    cudaFree(d_TAU);
    cudaFree(devInfo);
    cudaFree(work);
}
static double polyfit(double *x, double *y, double *p, int n){
    double *A = mat::createDoubleHost(3 * n);
    for(int i = 0; i < n; i++){
        A[i] = x[i] * x[i];
        A[i + n] = x[i];
        A[i + n * 2] = 1;
    }
    solveQR(A, y, p, n, 3);
    double rss = 0;
    for(int i = 0; i < n; i++){
        double ei = p[0] * x[i] * x[i] + p[1] * x[i] + p[2];
        rss += pow(y[i] - ei, 2);
    }
    return rss;
}
static float polyfit(float *fx, float *fy, float *fp, int n){
    double *x = mat::createDoubleHost(n);
    double *y = mat::createDoubleHost(n);
    double *p = mat::createDoubleHost(3);
    for(int i = 0; i < n; i++){
        x[i] = fx[i];
        y[i] = fy[i];
    }
    float rss = polyfit(x, y, p, n);
    for(int i = 0; i < 3; i++){
        fp[i] = p[i];
    }
    free(x);
    free(y);
    free(p);
    return rss;
}
static float str2float(const char *str){
    char str1[20] = {'\0'};
    char str2[20] = {'\0'};
    char str3[10] = {'\0'};

    float num = 0;

    int len = strlen(str);
    int offset = 0;
    char *current = str1;
    for(int i = 0; i < len; i++){
        if((str[i] >= 48 && str[i] <= 57) || str[i] == '+' || str[i] == '-'){
            current[i - offset] = str[i];
        }
        else if(str[i] == 'd' || str[i] == 'e'){
            offset = i + 1;
            current = str3;
        }
        else if(str[i] == '.'){
            offset = i;
            str2[0] = '.';
            current = str2;
        }
        else{
            break;
        }
    }

    float e = 1;
    float nege = 1;
    if(strlen(str3) > 0){
        int numi = atoi(str3);
        if(numi < 0){
            for(int i = 0; i < -numi; i++){
                nege *= 10;
            }
        }
        else{
            for(int i = 0; i < numi; i++){
                e *= 10;
            }
        }
    }
    if(strlen(str1) > 0){
        num = e * atoi(str1);
    }
    if(strlen(str2) > 0){
        float numf = e * atof(str2);
        if(num >= 0){
            num += numf;
        }
        else{
            num -= numf;
        }
    }
    return num / nege;
}
static int str2int(const char *str){
    return lroundf(str2float(str));
}
static void printStat(int i, int b){
    if(dat::display_mode == 0) return;
    for(int a=i+1; a<=getTaskIndex(i)+1; a++){
        if(b >= 100){
            if(a < 10){
                printf("  task 00%d of %d\n", a, b);
                continue;
            }
            if(a < 100){
                printf("  task 0%d of %d\n", a, b);
                continue;
            }
        }
        else if(b >= 10){
            if(a < 10){
                printf("  task 0%d of %d\n", a, b);
                continue;
            }
        }
        printf("  task %d of %d\n", a, b);
    }
}
static int getFileLength(FILE *file){
    fseek (file, 0, SEEK_END);
    int length = ftell (file);
    fseek (file, 0, SEEK_SET);
    return length;
}
static void initialiseModel(const char *model_dir){
    int npt;
    char path[80];

    sprintf(path, "%s/proc000000_x.bin", model_dir);
    FILE *xfile = fopen(path,"rb");
    sprintf(path, "%s/proc000000_z.bin", model_dir);
    FILE *zfile = fopen(path,"rb");

    fread(&npt, sizeof(int), 1, xfile);
    fread(&npt, sizeof(int), 1, zfile);

    npt /= 4;

    float *xbuffer = mat::createHost(npt);
    float *zbuffer = mat::createHost(npt);

    fread(xbuffer, sizeof(float), npt, xfile);
    fread(zbuffer, sizeof(float), npt, zfile);

    dat::Lx = 0;
    dat::Lz = 0;
    for(int i = 0; i < npt; i++){
        if(xbuffer[i] > dat::Lx) dat::Lx = xbuffer[i];
        if(zbuffer[i] > dat::Lz) dat::Lz = zbuffer[i];
    }
    dat::nx = lroundf(sqrt(npt * dat::Lx / dat::Lz));
    dat::nz = lroundf(npt / dat::nx);
    dat::nx2 = dat::nx * ntask;

    // dat::nx *= 2;
    // dat::nz *= 2;

    free(xbuffer);
    free(zbuffer);

    fclose(xfile);
    fclose(zfile);
}
static void readFortran(const char *fname, int isrc){
    FILE *parfile = fopen(fname,"r");
    char key[80];
    char value[80];

    int stat = 0;
    int offset = 0;

    char c = 0;
    int i = 0;

    while(c != EOF){
        c = fgetc(parfile);
        switch(c){
            case '\0': case '\r': case '\t': case '\n': case EOF:{
                if(stat == 4){
                    value[i - offset] = '\0';
                    stat = 5;
                }
                if(stat == 5){
                    if(isrc < 0){
                        if(strcmp(key, "simulation_mode") == 0){
                            dat::simulation_mode = str2int(value);
                        }
                        else if(strcmp(key, "wave_propagation_type") == 0){
                            switch(str2int(value)){
                                case 0: dat::wave_propagation_sh = 1; dat::wave_propagation_psv = 0; break;
                                case 1: dat::wave_propagation_sh = 0; dat::wave_propagation_psv = 1; break;
                                case 2: dat::wave_propagation_sh = 1; dat::wave_propagation_psv = 1; break;
                            }
                        }
                        else if(strcmp(key, "nt") == 0){
                            dat::nt = str2int(value);
                        }
                        else if(strcmp(key, "dt") == 0){
                            dat::dt = str2float(value);
                        }
                        else if(strcmp(key, "obs_type") == 0){
                            dat::obs_type = str2int(value);
                        }
                        else if(strcmp(key, "ntask") == 0){
                            dat::ntask = str2int(value);
                        }
                        else if(strcmp(key, "misfit_type") == 0){
                            dat::misfit_type = str2int(value);
                        }
                        else if(strcmp(key, "obs_su") == 0){
                            dat::obs_su = str2int(value);
                        }
                        else if(strcmp(key, "src_file") == 0){
                            dat::src_file = str2int(value);
                        }
                        else if(strcmp(key, "rec_file") == 0){
                            dat::rec_file = str2int(value);
                        }
                        else if(strcmp(key, "src_type") == 0){
                            if(dat::src_file == 0 && nsrc > 0){
                                dat::src_type = mat::createIntHost(nsrc);
                                int type = str2int(value);
                                for(int i = 0; i < nsrc; i++){
                                    dat::src_type[i] = type;
                                }
                            }
                        }
                        else if(strcmp(key, "src_f0") == 0){
                            if(dat::src_file == 0 && nsrc > 0){
                                dat::src_f0 = mat::createHost(nsrc);
                                mat::initHost(dat::src_f0, str2float(value), nsrc);
                            }
                        }
                        else if(strcmp(key, "src_t0") == 0){
                            if(dat::src_file == 0 && nsrc > 0){
                                dat::src_t0 = mat::createHost(nsrc);
                                mat::initHost(dat::src_t0, str2float(value), nsrc);
                            }
                        }
                        else if(strcmp(key, "src_angle") == 0){
                            if(dat::src_file == 0 && nsrc > 0){
                                dat::src_angle = mat::createHost(nsrc);
                                mat::initHost(dat::src_angle, str2float(value), nsrc);
                            }
                        }
                        else if(strcmp(key, "src_factor") == 0){
                            if(dat::src_file == 0 && nsrc > 0){
                                dat::src_factor = mat::createHost(nsrc);
                                mat::initHost(dat::src_factor, str2float(value), nsrc);
                            }
                        }
                        else if(strcmp(key, "src_align") == 0){
                            dat::src_align = str2int(value);
                        }
                        else if(strcmp(key, "rec_align") == 0){
                            dat::rec_align = str2int(value);
                        }
                        else if(strcmp(key, "absorb_left") == 0){
                            dat::absorb_left = str2int(value);
                        }
                        else if(strcmp(key, "absorb_right") == 0){
                            dat::absorb_right = str2int(value);
                        }
                        else if(strcmp(key, "absorb_top") == 0){
                            dat::absorb_top = str2int(value);
                        }
                        else if(strcmp(key, "absorb_bottom") == 0){
                            dat::absorb_bottom = str2int(value);
                        }
                        else if(strcmp(key, "absorb_width") == 0){
                            dat::absorb_width = str2float(value);
                        }
                        else if(strcmp(key, "nsrc") == 0){
                            dat::nsrc = str2int(value);
                        }
                        else if(strcmp(key, "nrec") == 0){
                            dat::nrec = str2int(value);
                        }
                        else if(strcmp(key, "sfe") == 0){
                            dat::sfe = str2int(value);
                        }
                        else if(strcmp(key, "filter_kernel") == 0){
                            dat::filter_kernel = str2int(value);
                        }
                        else if(strcmp(key, "unsharp_mask") == 0){
                            dat::unsharp_mask = str2int(value);
                        }
                        else if(strcmp(key, "inv_iteration") == 0){
                            dat::inv_iteration = str2int(value);
                        }
                        else if(strcmp(key, "inv_maxiter") == 0){
                            dat::inv_maxiter = str2int(value);
                        }
                        else if(strcmp(key, "lbfgs_mem") == 0){
                            dat::lbfgs_mem = str2int(value);
                        }
                        else if(strcmp(key, "optimize") == 0){
                            dat::optimize = str2int(value);
                        }
                        else if(strcmp(key, "ls_steplenmax") == 0){
                            dat::ls_steplenmax = str2float(value);
                        }
                        else if(strcmp(key, "ls_stepleninit") == 0){
                            dat::ls_stepleninit = str2float(value);
                        }
                        else if(strcmp(key, "ls_thresh") == 0){
                            dat::ls_thresh = str2float(value);
                        }
                        else if(strcmp(key, "ls_stepcountmax") == 0){
                            dat::ls_stepcountmax = str2int(value);
                        }
                        else if(strcmp(key, "parametrisation") == 0){
                            dat::parametrisation = str2int(value);
                        }
                        else if(strcmp(key, "model_init") == 0){
                            free(dat::model_init);
                            dat::model_init = copyString(value);
                        }
                        else if(strcmp(key, "model_true") == 0){
                            free(dat::model_true);
                            dat::model_true = copyString(value);
                        }
                        else if(strcmp(key, "output_path") == 0){
                            free(dat::output_path);
                            dat::output_path = copyString(value);
                        }
                        else if(strcmp(key, "obs_su_path") == 0){
                            free(dat::obs_su_path);
                            dat::obs_su_path = copyString(value);
                        }
                        else if(strcmp(key, "src_file_path") == 0){
                            free(dat::src_file_path);
                            dat::src_file_path = copyString(value);
                        }
                        else if(strcmp(key, "rec_file_path") == 0){
                            free(dat::rec_file_path);
                            dat::rec_file_path = copyString(value);
                        }
                        else if(strcmp(key, "inv_parameter") == 0){
                            dat::inv_parameter = str2int(value);
                        }
                        else if(strcmp(key, "display_mode") == 0){
                            dat::display_mode = str2int(value);
                        }
                    }
                    else{
                        if(strcmp(key, "xs") == 0){
                            dat::src_x[isrc] = str2float(value);
                        }
                        else if(strcmp(key, "zs") == 0){
                            dat::src_z[isrc] = str2float(value);
                        }
                        else if(strcmp(key, "f0") == 0){
                            dat::src_f0[isrc] = str2float(value);
                        }
                        else if(strcmp(key, "t0") == 0 || strcmp(key, "tshift") == 0){
                            dat::src_t0[isrc] = str2float(value);
                        }
                        else if(strcmp(key, "angle") == 0 || strcmp(key, "anglesource") == 0){
                            dat::src_angle[isrc] = str2float(value);
                        }
                        else if(strcmp(key, "factor") == 0){
                            dat::src_factor[isrc] = str2float(value);
                        }
                        else if(strcmp(key, "type") == 0 || strcmp(key, "time_function_type") == 0){
                            dat::src_type[isrc] = str2float(value);
                        }
                    }
                }
                stat = 0;
                offset = 0;
                i = -1;
                break;
            }
            case '#':{
                switch(stat){
                    case 4: value[i - offset] = '\0'; stat = 5; break;
                    case 5: break;
                    default: stat = -1;
                }
                break;
            }
            case ' ':{
                switch(stat){
                    case 1: key[i - offset] = '\0'; stat = 2; break;
                    case 4: value[i - offset] = '\0'; stat = 5; break;
                }
                break;
            }
            case '=':{
                switch(stat){
                    case 1: key[i - offset] = '\0'; stat = 3; break;
                    case 2: stat = 3; break;
                    case 5: break;
                    default: stat = -1;
                }
                break;
            }
            default:{
                if(c >= 65 && c <= 90){
                    c += 32;
                }
                switch(stat){
                    case 0: stat = 1; offset = i; key[0] = c; break;
                    case 1: key[i - offset] = c; break;
                    case 2: stat = -1; break;
                    case 3: stat = 4; offset = i; value[0] = c; break;
                    case 4: value[i - offset] = c; break;
                }
            }
        }
        i++;
    }

    fclose(parfile);
}
static int loadModel(const char *model_dir, float mp, float ms, float mr){
    char path[80];

    sprintf(path, "%s/proc000000_x.bin", model_dir);
    FILE *xfile = fopen(path,"rb");
    sprintf(path, "%s/proc000000_z.bin", model_dir);
    FILE *zfile = fopen(path,"rb");
    sprintf(path, "%s/proc000000_rho.bin", model_dir);
    FILE *rfile = fopen(path,"rb");
    sprintf(path, "%s/proc000000_vp.bin", model_dir);
    FILE *pfile = fopen(path,"rb");
    sprintf(path, "%s/proc000000_vs.bin", model_dir);
    FILE *sfile = fopen(path,"rb");

    int npt;
    fread(&npt, sizeof(int), 1, xfile);
    fread(&npt, sizeof(int), 1, zfile);
    fread(&npt, sizeof(int), 1, rfile);
    fread(&npt, sizeof(int), 1, pfile);
    fread(&npt, sizeof(int), 1, sfile);

    npt /= 4;

    float *xbuffer = mat::createHost(npt);
    float *zbuffer = mat::createHost(npt);
    float *rbuffer = mat::createHost(npt);
    float *pbuffer = mat::createHost(npt);
    float *sbuffer = mat::createHost(npt);

    fread(xbuffer, sizeof(float), npt, xfile);
    fread(zbuffer, sizeof(float), npt, zfile);
    fread(rbuffer, sizeof(float), npt, rfile);
    fread(pbuffer, sizeof(float), npt, pfile);
    fread(sbuffer, sizeof(float), npt, sfile);

    if(mp > 0) mat::initHost(pbuffer, mp, npt);
    if(ms > 0) mat::initHost(sbuffer, ms, npt);
    if(mr > 0) mat::initHost(rbuffer, mr, npt);

    float *dxbuffer = mat::create(npt);
    float *dzbuffer = mat::create(npt);
    float *drbuffer = mat::create(npt);
    float *dpbuffer = mat::create(npt);
    float *dsbuffer = mat::create(npt);

    mat::copyHostToDevice(dxbuffer, xbuffer, npt);
    mat::copyHostToDevice(dzbuffer, zbuffer, npt);
    mat::copyHostToDevice(drbuffer, rbuffer, npt);
    mat::copyHostToDevice(dpbuffer, pbuffer, npt);
    mat::copyHostToDevice(dsbuffer, sbuffer, npt);

    float dmax = dat::Lx * dat::Lx + dat::Lz * dat::Lz;
    mesh2grid<<<nxb, nzt>>>(dxbuffer, dzbuffer, drbuffer, dpbuffer, dsbuffer,
        dat::lambda, dat::mu, dat::rho, dat::Lx/(nx-1), dat::Lz/(nz-1), dmax, npt);
    if(dat::parametrisation){
        changeParametrisation<<<nxb, nzt>>>(dat::lambda, dat::mu, dat::rho, psv);
    }

    free(xbuffer);
    free(zbuffer);
    free(rbuffer);
    free(pbuffer);
    free(sbuffer);

    cudaFree(dxbuffer);
    cudaFree(dzbuffer);
    cudaFree(drbuffer);
    cudaFree(dpbuffer);
    cudaFree(dsbuffer);

    fclose(xfile);
    fclose(zfile);
    fclose(rfile);
    fclose(pfile);
    fclose(sfile);

    return 1;
}
static int loadModel(const char *model_dir){
    return loadModel(model_dir, -1, -1, -1);
}
static void initialisePosition(float *d_pos_x, float *d_pos_z, int n, int type){
    float &width = dat::absorb_width;
    if(type == 4){
        float *posx = mat::createHost(n);
        float *posz = mat::createHost(n);

        float nxf = sqrt((float)n * dat::nx / dat::nz);
        float nzf = (float)n / nxf;
        int nx = lroundf(ceil(nxf));
        int nz = lroundf(ceil(nzf));

        float Sx = 0;
        float Sz = 0;
        float Lx = dat::Lx;
        float Lz = dat::Lz;

        if(dat::absorb_left){
            Lx -= width;
            Sx = width;
        }
        if(dat::absorb_right){
            Lx -= width;
        }
        float dx = Lx / (nx + 1);

        if(dat::absorb_top){
            Lz -= width;
            Sz = width;
        }
        if(dat::absorb_bottom){
            Lz -= width;
        }
        float dz = Lz / (nz + 1);

        int k = 0;
        for(int i = 0; i < nx; i++){
            for(int j = 0; j < nz; j++){
                if(k >= n) break;
                posx[k] = Sx + dx * (i + 1);
                posz[k] = Sz + dz * (j + 1);
                k++;
            }
        }

        mat::copyHostToDevice(d_pos_x, posx, n);
        mat::copyHostToDevice(d_pos_z, posz, n);
    }
    else{
        float dx = 2 * dat::Lx / (nx - 1);
        float dz = 2 * dat::Lz / (nz - 1);
        float *pos = mat::createHost(n);
        switch(type){
            case 0:{
                if(dat::absorb_top){
                    mat::init(d_pos_z, width, n);
                }
                else{
                    mat::init(d_pos_z, dz, n);
                }
                break;
            }
            case 1:{
                if(dat::absorb_bottom){
                    mat::init(d_pos_z, dat::Lz - width, n);
                }
                else{
                    mat::init(d_pos_z, dat::Lz - dz, n);
                }
                break;
            }
            case 2:{
                if(dat::absorb_left){
                    mat::init(d_pos_x, width, n);
                }
                else{
                    mat::init(d_pos_x, dx, n);
                }
                break;
            }
            case 3:{
                if(dat::absorb_right){
                    mat::init(d_pos_x, dat::Lx - width, n);
                }
                else{
                    mat::init(d_pos_x, dat::Lx - dx, n);
                }
                break;
            }
        }
        float L, d;
        float S = 0;
        if(type == 0 || type == 1){
            L = dat::Lx;
            if(dat::absorb_left){
                L -= width;
                S = width;
            }
            if(dat::absorb_right){
                L -= width;
            }
            d = L / (n + 1);
            for(int i = 0; i < n; i++){
                pos[i] = S + d * (i + 1);
            }
            mat::copyHostToDevice(d_pos_x, pos, n);
        }
        else if(type == 2 || type == 3){
            L = dat::Lz;
            if(dat::absorb_top){
                L -= width;
                S = width;
            }
            if(dat::absorb_bottom){
                L -= width;
            }
            d = L / (n + 1);
            for(int i = 0; i < n; i++){
                pos[i] = S + d * (i + 1);
            }
            mat::copyHostToDevice(d_pos_z, pos, n);
        }
        free(pos);
    }

}
static int importData(const char *datapath, int argc, const char *argv[]){
    dat::wave_propagation_sh = 1;
    dat::wave_propagation_psv = 0;
    dat::obs_type = 0;
    dat::ntask = 1;
    dat::misfit_type = 0;
    dat::parametrisation = 1;
    dat::obs_su = 0;
    dat::src_file = 0;
    dat::rec_file = 0;
    dat::nt = 5000;
    dat::dt = 0.06;
    dat::sfe = 10;
    dat::misfit_ref = 1;

    dat::absorb_bottom = 1;
    dat::absorb_right = 1;
    dat::absorb_top = 1;
    dat::absorb_left = 1;
    dat::absorb_width = 48000;

    dat::obs_su_path = copyString("trace");
    dat::src_file_path = copyString("data");
    dat::rec_file_path = copyString("data");
    dat::optimize = 1;
    dat::filter_kernel = 4;
    dat::unsharp_mask = 0;
    dat::inv_maxiter = 0;
    dat::lbfgs_mem = 5;
    dat::ls_stepleninit = 0.05;
    dat::ls_steplenmax = 0.5;
    dat::ls_stepcountmax = 10;
    dat::ls_thresh = 1.2;
    dat::inv_parameter = 1;

    char path[80];
    readFortran(datapath, -1);

    if(argc > 1){
        strcat(dat::argstr, "# ------------ argv ------------\n");
        for(int i = 1; i < argc; i += 2){
            if(strlen(argv[i]) == 2){
                switch(argv[i][1]){
                    case 'o':{
                        strcat(dat::argstr, "output_path                     = ");
                        dat::output_path = copyString(argv[i+1]);
                        break;
                    }
                    case 'm':{
                        strcat(dat::argstr, "simulation_mode                 = ");
                        dat::simulation_mode = str2int(argv[i+1]);
                        break;
                    }
                    case 'p':{
                        strcat(dat::argstr, "display_mode                    = ");
                        dat::display_mode = str2int(argv[i+1]);
                        break;
                    }
                    case 'c':{
                        strcat(dat::argstr, "par_file                        = ");
                        dat::display_mode = str2int(argv[i+1]);
                        break;
                    }
                }
            }
            else if(strlen(argv[i]) == 3){
                switch(argv[i][1]){
                    case 'm':{
                        switch(argv[i][2]){
                            case 't':{
                                strcat(dat::argstr, "model_true                      = ");
                                dat::model_true = copyString(argv[i+1]);
                                break;
                            }
                            case 'i':{
                                strcat(dat::argstr, "model_init                      = ");
                                dat::model_init = copyString(argv[i+1]);
                                break;
                            }
                            case 'p':{
                                strcat(dat::argstr, "model_init_vp                   = ");
                                dat::model_p = str2float(argv[i+1]);
                                break;
                            }
                            case 's':{
                                strcat(dat::argstr, "model_init_vs                   = ");
                                dat::model_s = str2float(argv[i+1]);
                                break;
                            }
                            case 'r':{
                                strcat(dat::argstr, "model_init_rho                  = ");
                                dat::model_r = str2float(argv[i+1]);
                                break;
                            }
                        }
                        break;
                    }
                    case 'n':{
                        switch(argv[i][2]){
                            case 'i':{
                                strcat(dat::argstr, "inv_iteration                   = ");
                                dat::inv_iteration = str2int(argv[i+1]);
                                break;
                            }
                            case 's':{
                                strcat(dat::argstr, "nsrc                            = ");
                                dat::nsrc = str2int(argv[i+1]);
                                break;
                            }
                            case 'r':{
                                strcat(dat::argstr, "nrec                            = ");
                                dat::nrec = str2int(argv[i+1]);
                                break;
                            }
                        }
                        break;
                    }
                    case 'a':{
                        switch(argv[i][2]){
                            case 's':{
                                strcat(dat::argstr, "src_align                       = ");
                                dat::src_align = str2int(argv[i+1]);
                                break;
                            }
                            case 'r':{
                                strcat(dat::argstr, "rec_align                       = ");
                                dat::rec_align = str2int(argv[i+1]);
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            else {
                break;
            }
            strcat(dat::argstr, argv[i+1]);
            strcat(dat::argstr, "\n");
        }
        strcat(dat::argstr, "\n# ------------ file ------------");
    }

    initialiseModel(dat::model_init);
    createDirectory(dat::output_path);

    if(dat::src_file){
        dat::src_x = mat::createHost(nsrc);
        dat::src_z = mat::createHost(nsrc);
        dat::src_type = mat::createIntHost(nsrc);
        dat::src_f0 = mat::createHost(nsrc);
        dat::src_t0 = mat::createHost(nsrc);
        dat::src_angle = mat::createHost(nsrc);
        dat::src_factor = mat::createHost(nsrc);

        for(int isrc = 0; isrc < nsrc; isrc++){
            if(isrc < 10){
                sprintf(path, "%s/SOURCE_00000%d", dat::src_file_path, isrc);
            }
            else if(isrc < 100){
                sprintf(path, "%s/SOURCE_0000%d", dat::src_file_path, isrc);
            }
            else{
                sprintf(path, "%s/SOURCE_000%d", dat::src_file_path, isrc);
            }
            readFortran(path, isrc);
        }

        float *src_x = dat::src_x;
        float *src_z = dat::src_z;

        dat::src_x = mat::create(nsrc);
        dat::src_z = mat::create(nsrc);

        mat::copyHostToDevice(dat::src_x, src_x, nsrc);
        mat::copyHostToDevice(dat::src_z, src_z, nsrc);

        free(src_x);
        free(src_z);
    }
    else{
        dat::src_x = mat::create(nsrc);
        dat::src_z = mat::create(nsrc);
        initialisePosition(dat::src_x, dat::src_z, nsrc, dat::src_align);
    }

    if(dat::rec_file){
        sprintf(path, "%s/STATIONS", dat::rec_file_path);
        FILE *stfile = fopen(path,"r");
        char buffer[80];
        char numbuffer[40];

        // dat::nrec = 0;
        // while(fgets(buffer, 80, stfile) != NULL){
        //     if(buffer[0] == 'S'){
        //         dat::nrec ++;
        //     }
        // }
        // fseek (stfile, 0, SEEK_SET);

        float *rec_x = mat::createHost(nrec);
        float *rec_z = mat::createHost(nrec);

        int irec = 0;
        while(fgets(buffer, 80, stfile) != NULL){
            if(buffer[0] == 'S'){
                int stat = 0;
                int offset = 0;
                for(int i = 0; i < 80 && buffer[i] != '\0'; i++){
                    if(buffer[i] == ' '){
                        switch(stat){
                            case 0: stat++; break;
                            case 2: stat++; break;
                            case 4:{
                                stat++;
                                numbuffer[i - offset] = '\0';
                                rec_x[irec] = str2float(numbuffer);
                                break;
                            }
                            case 6:{
                                stat++;
                                numbuffer[i - offset] = '\0';
                                rec_z[irec] = str2float(numbuffer);
                                i = 80;
                                break;
                            }
                        }
                    }
                    else{
                        if(stat == 1 || stat == 3 || stat == 5){
                            stat++;
                            offset = i;
                        }
                        if(stat == 4 || stat == 6){
                            numbuffer[i - offset] = buffer[i];
                        }
                    }
                }
                irec++;
            }
        }
        dat::rec_x = mat::create(nrec);
        dat::rec_z = mat::create(nrec);

        mat::copyHostToDevice(dat::rec_x, rec_x, nrec);
        mat::copyHostToDevice(dat::rec_z, rec_z, nrec);

        free(rec_x);
        free(rec_z);

        fclose(stfile);
    }
    else{
        dat::rec_x = mat::create(nrec);
        dat::rec_z = mat::create(nrec);
        initialisePosition(dat::rec_x, dat::rec_z, nrec, dat::rec_align);
    }

    {
        int adjoint = (dat::simulation_mode != 1);
        dat::nxb = dim3(nx, 1);
        dat::nxb2 = dim3(nx, ntask);
        dat::nzt = dim3(nz);
        dat::isrc = mat::createIntHost(2);

        dat::X = mat::create(nx, nz);
        dat::Z = mat::create(nx, nz);
        initialiseGrids<<<nxb, nzt>>>(dat::X, dat::Z, dat::Lx, nx, dat::Lz, nz);

        if(nt % dat::sfe != 0){
            nt = dat::sfe * lroundf((float)nt / dat::sfe);
        }
        dat::nsfe = nt / dat::sfe;

        if(dat::absorb_width < 1){
            dat::absorb_left = 0;
            dat::absorb_right = 0;
            dat::absorb_top = 0;
            dat::absorb_bottom = 0;
        }


        if(sh){
            dat::vy = mat::create(nx2, nz);
            dat::uy = mat::create(nx2, nz);
            dat::sxy = mat::create(nx2, nz);
            dat::szy = mat::create(nx2, nz);
            dat::dsy = mat::create(nx2, nz);
            dat::dvydx = mat::create(nx2, nz);
            dat::dvydz = mat::create(nx2, nz);

            dat::out_y = mat::create(nrec, nt * ntask);
            dat::uy_forward = mat::createHost(dat::nsfe, nx2, nz);
            dat::vy_forward = mat::createHost(dat::nsfe, nx2, nz);
        }
        if(psv){
            dat::vx = mat::create(nx2, nz);
            dat::vz = mat::create(nx2, nz);
            dat::ux = mat::create(nx2, nz);
            dat::uz = mat::create(nx2, nz);
            dat::sxx = mat::create(nx2, nz);
            dat::szz = mat::create(nx2, nz);
            dat::sxz = mat::create(nx2, nz);
            dat::dsx = mat::create(nx2, nz);
            dat::dsz = mat::create(nx2, nz);
            dat::dvxdx = mat::create(nx2, nz);
            dat::dvxdz = mat::create(nx2, nz);
            dat::dvzdx = mat::create(nx2, nz);
            dat::dvzdz = mat::create(nx2, nz);

            dat::out_x = mat::create(nrec, nt * ntask);
            dat::out_z = mat::create(nrec, nt * ntask);
            dat::ux_forward = mat::createHost(dat::nsfe, nx2, nz);
            dat::uz_forward = mat::createHost(dat::nsfe, nx2, nz);
            dat::vx_forward = mat::createHost(dat::nsfe, nx2, nz);
            dat::vz_forward = mat::createHost(dat::nsfe, nx2, nz);
        }

        dat::lambda = mat::create(nx, nz);
        dat::rho = mat::create(nx, nz);
        dat::mu = mat::create(nx, nz);
        dat::absbound = mat::create(nx, nz);

        dat::stf_x = mat::create(nsrc, nt);
        dat::stf_y = mat::create(nsrc, nt);
        dat::stf_z = mat::create(nsrc, nt);

        if(adjoint){
            if(sh){
                dat::dvydx_fw = mat::create(nx2, nz);
                dat::dvydz_fw = mat::create(nx2, nz);

                dat::u_obs_y = mat::create(nsrc, nrec, nt);
            }
            if(psv){
                dat::dvxdx_fw = mat::create(nx2, nz);
                dat::dvxdz_fw = mat::create(nx2, nz);
                dat::dvzdx_fw = mat::create(nx2, nz);
                dat::dvzdz_fw = mat::create(nx2, nz);

                dat::u_obs_x = mat::create(nsrc, nrec, nt);
                dat::u_obs_z = mat::create(nsrc, nrec, nt);
            }

            dat::K_lambda = mat::create(nx, nz);
            dat::K_mu = mat::create(nx, nz);
            dat::K_rho = mat::create(nx, nz);

            dat::adstf_x = mat::create(nrec, nt*ntask);
            dat::adstf_y = mat::create(nrec, nt*ntask);
            dat::adstf_z = mat::create(nrec, nt*ntask);
        }

        dat::src_x_id = mat::createInt(nsrc);
        dat::src_z_id = mat::createInt(nsrc);
        dat::rec_x_id = mat::createInt(nrec);
        dat::rec_z_id = mat::createInt(nrec);

        computeIndices<<<nsrc, 1>>>(dat::src_x_id, dat::src_x, dat::Lx, nx);
        computeIndices<<<nsrc, 1>>>(dat::src_z_id, dat::src_z, dat::Lz, nz);
        computeIndices<<<nrec, 1>>>(dat::rec_x_id, dat::rec_x, dat::Lx, nx);
        computeIndices<<<nrec, 1>>>(dat::rec_z_id, dat::rec_z, dat::Lz, nz);

        initialiseAbsorbingBoundaries<<<nxb, nzt>>>(
            dat::absbound, dat::absorb_width,
            dat::absorb_left, dat::absorb_right, dat::absorb_bottom, dat::absorb_top,
            dat::Lx, dat::Lz, dat::X, dat::Z
        );
    }

    return 1;
}
static void exportData(int iter){
    iter++;
    int kernel = (iter > 0 && iter <= dat::inv_iteration);

    char name[80];
    if(iter < 10){
        sprintf(name, "%s/000%d", dat::output_path, iter);
    }
    else if(iter < 100){
        sprintf(name, "%s/00%d", dat::output_path, iter);
    }
    else if(iter < 1000){
        sprintf(name, "%s/0%d", dat::output_path, iter);
    }
    else{
        sprintf(name, "%s/%d", dat::output_path, iter);
    }

    createDirectory(name);

    char path[80];
    sprintf(path, "%s/proc000000_x.bin", name);
    FILE *xfile = fopen(path,"wb");
    sprintf(path, "%s/proc000000_z.bin", name);
    FILE *zfile = fopen(path,"wb");
    sprintf(path, "%s/proc000000_rho.bin", name);
    FILE *rfile = fopen(path,"wb");
    sprintf(path, "%s/proc000000_vp.bin", name);
    FILE *pfile = fopen(path,"wb");
    sprintf(path, "%s/proc000000_vs.bin", name);
    FILE *sfile = fopen(path,"wb");

    FILE *krfile = NULL;
    FILE *klfile = NULL;
    FILE *kmfile = NULL;
    if(kernel){
        sprintf(path, "%s/kernel_rho.bin", name);
        krfile = fopen(path,"wb");
        sprintf(path, "%s/kernel_lambda.bin", name);
        klfile = fopen(path,"wb");
        sprintf(path, "%s/kernel_mu.bin", name);
        kmfile = fopen(path,"wb");
    }

    int npt = nx * nz * 4;
    fwrite(&npt, sizeof(int), 1, xfile);
    fwrite(&npt, sizeof(int), 1, zfile);
    fwrite(&npt, sizeof(int), 1, rfile);
    fwrite(&npt, sizeof(int), 1, pfile);
    fwrite(&npt, sizeof(int), 1, sfile);

    if(kernel){
        fwrite(&npt, sizeof(int), 1, krfile);
        fwrite(&npt, sizeof(int), 1, kmfile);
        fwrite(&npt, sizeof(int), 1, klfile);
    }
    npt /= 4;

    float *buffer = mat::createHost(npt);
    mat::copyDeviceToHost(buffer, mat::getDataPointer(dat::X), npt);
    fwrite(buffer, sizeof(float), npt, xfile);
    mat::copyDeviceToHost(buffer, mat::getDataPointer(dat::Z), npt);
    fwrite(buffer, sizeof(float), npt, zfile);
    mat::copyDeviceToHost(buffer, mat::getDataPointer(dat::rho), npt);
    fwrite(buffer, sizeof(float), npt, rfile);

    float *vp = mat::create(npt);
    mat::copy(vp, mat::getDataPointer(dat::lambda), npt);
    float *vs = mat::create(npt);
    mat::copy(vs, mat::getDataPointer(dat::mu), npt);
    float *rho = mat::create(npt);
    mat::copy(rho, mat::getDataPointer(dat::rho), npt);
    if(dat::parametrisation){
        changeParametrisation<<<nxb, nzt>>>(vp, vs, rho, nz, psv);
    }

    mat::copyDeviceToHost(buffer, vp, npt);
    fwrite(buffer, sizeof(float), npt, pfile);
    mat::copyDeviceToHost(buffer, vs, npt);
    fwrite(buffer, sizeof(float), npt, sfile);

    if(kernel){
        mat::copyDeviceToHost(buffer, mat::getDataPointer(dat::K_rho), npt);
        fwrite(buffer, sizeof(float), npt, krfile);
        mat::copyDeviceToHost(buffer, mat::getDataPointer(dat::K_mu), npt);
        fwrite(buffer, sizeof(float), npt, kmfile);
        mat::copyDeviceToHost(buffer, mat::getDataPointer(dat::K_lambda), npt);
        fwrite(buffer, sizeof(float), npt, klfile);
    }

    cudaFree(vp);
    cudaFree(vs);
    cudaFree(rho);
    free(buffer);

    fclose(xfile);
    fclose(zfile);
    fclose(rfile);
    fclose(pfile);
    fclose(sfile);

    if(kernel){
        fclose(krfile);
        fclose(kmfile);
        fclose(klfile);
    }
}
static void checkMemoryUsage(){
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;
    float free_db = (float)free_byte ;
    float total_db = (float)total_byte ;
    float used_db = total_db - free_db ;

    printf("Memory usage: %.1fMB / %.1fMB\n", used_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}
static void applyGaussian(float **p, int sigma){
    float **gsum = mat::create(nx, nz);
    float **gtemp = mat::create(nx, nz);
    initialiseGaussian<<<nxb, nzt>>>(gsum, nx, nz, sigma);
    filterKernelX<<<nxb, nzt>>>(p, gtemp, nx, sigma);
    filterKernelZ<<<nxb, nzt>>>(p, gtemp, gsum, nz, sigma);
    mat::freeDevice(gsum);
    mat::freeDevice(gtemp);
}
static void applyUnsharpMask(float **p, int sigma, int n){
    float **p2 = mat::create(nx, nz);
    for(int i=0; i<sigma; i++){
        mat::copy(p2, p, nx, nz);
        applyGaussian(p2, n);
        mat::calc(p, 2, p, -1, p2, nx, nz);
    }
    mat::freeDevice(p2);
}
static void generateChecker(float **p, float dp, float margin, int cx, int cz){
    float lx = dat::Lx / (cx + (cx + 3) * margin);
    float lz = dat::Lz / (cz + (cz + 3) * margin);
    generateChecker<<<nxb, nzt>>>(p, dp, margin, lx, lz, dat::X, dat::Z);

    int sigma = (int) 15 / cx;
    applyGaussian(p, sigma);
}
static void generateLayer(float **p, float dp, int n){
    float dz = dat::Lz / n;
    float dpi = 2*dp / (n-1);
    for(int i = 0; i < n; i++){
        generateLayer<<<nxb, nzt>>>(p, i*dz, (i+1)*dz, -dp + dpi*i, dat::Z);
    }

    int sigma = (int) 15 / n;
    applyGaussian(p, sigma);
}
static void generateRandomLayer(float **p, float dp, float dl, int n){
    float dz = dat::Lz / n;
    float dpi = 2*dp / (n-1);
    float *layer1 = mat::create(nx);
    float *layer2 = mat::create(nx);
    float *hlayer = mat::createHost(nx);
    float base = dl * dz / n;
    float dx = dat::Lx / (nx - 1);
    srand(time(0));
    for(int i = 0; i < n; i++){
        mat::initHost(hlayer, 0, nx);
        for(int k = 0; k < n; k++){
            float rng = (float)(rand() % 101) / 100;
            for(int j = 0; j < nx; j++){
                hlayer[j] += base * sin((k+rng)*2*pi*j*dx/dat::Lx+rng*pi);
            }
        }
        mat::copyHostToDevice(layer2, hlayer, nx);
        if(i==0){
            mat::init(layer1,0,nx);
        }
        else if(i==n-1){
            mat::init(layer2,0,nx);
        }
        generateRandomLayer<<<nxb, nzt>>>(p, i*dz, (i+1)*dz, -dp + dpi*i, layer1, layer2, dat::Z);
        mat::copy(layer1, layer2, nx);
    }

    int sigma = (int) 12 / n;
    applyGaussian(dat::mu, sigma);

    cudaFree(layer1);
    cudaFree(layer2);
    free(hlayer);
}
static void writeSU(const char *fname, const int isrc, float **data){
    FILE *su = fopen(fname, "wb");

    int header1[28];
    short int header2[2];
    short int header3[2];
    float header4[30];

    for(int i = 0; i < 28; i++) header1[i] = 0;
    for(int i = 0; i < 2; i++) header2[i] = 0;
    for(int i = 0; i < 2; i++) header3[i] = 0;
    for(int i = 0; i < 30; i++) header4[i] = 0;

    float *src_x = mat::createHost(nsrc);
    float *src_z = mat::createHost(nsrc);
    mat::copyDeviceToHost(src_x, dat::src_x, nsrc);
    mat::copyDeviceToHost(src_z, dat::src_z, nsrc);

    float xs = src_x[isrc];
    float zs = src_z[isrc];

    free(src_x);
    free(src_z);

    float *rec_x = mat::createHost(nrec);
    float *rec_z = mat::createHost(nrec);
    mat::copyDeviceToHost(rec_x, dat::rec_x, nsrc);
    mat::copyDeviceToHost(rec_z, dat::rec_z, nsrc);

    short int dt_int2;
    if(dt * 1e6 > pow(2, 15)){
        dt_int2 = 0;
    }
    else{
        dt_int2 = (short int)(dt * 1e6);
    }

    header1[18] = lroundf(xs);
    header1[19] = lroundf(zs);

    header2[0] = 0;
    header2[1] = nt;

    header3[0] = dt_int2;
    header3[1] = 0;

    for(int irec = 0; irec < nrec; irec++){
        header1[0] = irec + 1;
        header1[9] = lroundf(rec_x[irec] - xs);
        header1[20] = lroundf(rec_x[irec]);
        header1[21] = lroundf(rec_z[irec]);

        if(nrec > 1){
            header4[1] = rec_x[1] - rec_x[0];
        }

        fwrite(header1, sizeof(int), 28, su);
        fwrite(header2, sizeof(short int), 2, su);
        fwrite(header3, sizeof(short int), 2, su);
        fwrite(header4, sizeof(float), 30, su);
        fwrite(data[irec], sizeof(float), nt, su);
    }

    free(rec_x);
    free(rec_z);

    fclose(su);
}
static void writeSU(float ***u_obs, char c){
    char path[80];
    for(int i = 0; i < nsrc; i++){
        if(i < 10){
            sprintf(path, "%s/U%c_00000%d", dat::obs_su_path, c, i);
        }
        else if(i < 100){
            sprintf(path, "%s/U%c_0000%d", dat::obs_su_path, c, i);
        }
        else{
            sprintf(path, "%s/U%c_000%d", dat::obs_su_path, c, i);
        }
        writeSU(path, i, u_obs[i]);
    }
}
static void writeSU(){
    createDirectory(dat::obs_su_path);

    float ***u_obs = mat::createHost(nsrc, nrec, nt);
    if(sh){
        mat::copyDeviceToHost(u_obs, dat::u_obs_y, nsrc, nrec, nt);
        writeSU(u_obs, 'y');
    }
    if(psv){
        mat::copyDeviceToHost(u_obs, dat::u_obs_x, nsrc, nrec, nt);
        writeSU(u_obs, 'x');
        mat::copyDeviceToHost(u_obs, dat::u_obs_z, nsrc, nrec, nt);
        writeSU(u_obs, 'z');
    }
    mat::freeHost(u_obs);
}
static void readSU(const char *fname, float **data){
    FILE *su = fopen(fname, "rb");

    int header1[28];
    short int header2[2];
    short int header3[2];
    float header4[30];

    fread(header1, sizeof(int), 28, su);
    fread(header2, sizeof(short int), 2, su);

    int nt_su = header2[1];
    int nrec_su = getFileLength(su) / (240 + 4 * nt);

    if(nt_su != nt || nrec_su != nrec){
        printf("Error loading Seismic Unix file\n");
    }
    else{
        for(int irec = 0; irec < nrec; irec++){
            fread(header1, sizeof(int), 28, su);
            fread(header2, sizeof(short int), 2, su);
            fread(header3, sizeof(short int), 2, su);
            fread(header4, sizeof(float), 30, su);
            fread(data[irec], sizeof(float), nt, su);
        }
    }

    fclose(su);
}
static void readSU(float ***u_obs, char c){
    char path[80];
    for(int i = 0; i < nsrc; i++){
        if(i < 10){
            sprintf(path, "%s/U%c_00000%d", dat::obs_su_path, c, i);
        }
        else if(i < 100){
            sprintf(path, "%s/U%c_0000%d", dat::obs_su_path, c, i);
        }
        else{
            sprintf(path, "%s/U%c_000%d", dat::obs_su_path, c, i);
        }
        readSU(path, u_obs[i]);
    }
}
static void readSU(){
    float ***u_obs = mat::createHost(nsrc, nrec, nt);
    if(sh){
        readSU(u_obs, 'y');
        mat::copyHostToDevice(dat::u_obs_y, u_obs, nsrc, nrec, nt);
    }
    if(psv){
        readSU(u_obs, 'x');
        mat::copyHostToDevice(dat::u_obs_x, u_obs, nsrc, nrec, nt);
        readSU(u_obs, 'z');
        mat::copyHostToDevice(dat::u_obs_z, u_obs, nsrc, nrec, nt);
    }
    mat::freeHost(u_obs);
}
static void makeSourceTimeFunction(float *stf, int index){
    float max = 0;
    float f0 = dat::src_f0[index];
    float t0 = dat::src_t0[index];
    for(int it = 0; it < nt; it++){
        float t = it * dt;
        switch(dat::src_type[index]){
            case 1:{
                float a = pi * pi * f0 * f0;
                stf[it] = -(t - t0) * exp(-pow(a, 2) * pow(t - t0, 2));
                break;
            }
            // other stf: later
        }

        if(fabs(stf[it]) > max){
            max = fabs(stf[it]);
        }
    }
    if(max > 0){
        for(int it = 0; it < nt; it++){
            stf[it] /= max;
        }
    }
}
static void prepareSTF(){
    float **stf_x = mat::createHost(nsrc, nt);
    float **stf_y = mat::createHost(nsrc, nt);
    float **stf_z = mat::createHost(nsrc, nt);
    float *stfn = mat::createHost(nt);

    for(int isrc = 0; isrc < nsrc; isrc++){
        makeSourceTimeFunction(stfn, isrc);
        float angle = dat::src_angle[isrc];
        float amp = dat::src_factor[isrc];
        for(int it = 0; it < nt; it++){
            stf_x[isrc][it] = amp * stfn[it] * cos(angle);
            stf_y[isrc][it] = amp * stfn[it];
            stf_z[isrc][it] = amp * stfn[it] * sin(angle);
        }
    }

    mat::copyHostToDevice(dat::stf_x, stf_x, nsrc, nt);
    mat::copyHostToDevice(dat::stf_y, stf_y, nsrc, nt);
    mat::copyHostToDevice(dat::stf_z, stf_z, nsrc, nt);

    mat::freeHost(stf_x);
    mat::freeHost(stf_y);
    mat::freeHost(stf_z);
    free(stfn);
}
static void initialiseDynamicFields(){
    if(sh){
        mat::init(dat::vy, 0, nx2, nz);
        mat::init(dat::uy, 0, nx2, nz);
        mat::init(dat::sxy, 0, nx2, nz);
        mat::init(dat::szy, 0, nx2, nz);
    }
    if(psv){
        mat::init(dat::vx, 0, nx2, nz);
        mat::init(dat::vz, 0, nx2, nz);
        mat::init(dat::ux, 0, nx2, nz);
        mat::init(dat::uz, 0, nx2, nz);
        mat::init(dat::sxx, 0, nx2, nz);
        mat::init(dat::szz, 0, nx2, nz);
        mat::init(dat::sxz, 0, nx2, nz);
    }
}
static void initialiseKernels(){
    mat::init(dat::K_lambda, 0, nx, nz);
    mat::init(dat::K_mu, 0, nx, nz);
    mat::init(dat::K_rho, 0, nx, nz);
}
static void runWaveFieldPropagation(){
    initialiseDynamicFields();

    int ntask2 = dat::isrc[1]-dat::isrc[0]+1;
    for(int it = 0; it < nt; it++){
        if(mode == 0){
            if((it + 1) % dat::sfe == 0){
                int isfe = dat::nsfe - (it + 1) / dat::sfe;
                if(sh){
                    mat::copyDeviceToHost(dat::uy_forward[isfe], dat::uy, nx2, nz);
                }
                if(psv){
                    mat::copyDeviceToHost(dat::ux_forward[isfe], dat::ux, nx2, nz);
                    mat::copyDeviceToHost(dat::uz_forward[isfe], dat::uz, nx2, nz);
                }
            }
        }

        if(sh){
            divSY<<<nxb2, nzt>>>(dat::dsy, dat::sxy, dat::szy, dat::X, dat::Z, nx, nz);
        }
        if(psv){
            divSXZ<<<nxb2, nzt>>>(dat::dsx, dat::dsz, dat::sxx, dat::szz, dat::sxz, dat::X, dat::Z, nx, nz);
        }
        if(mode == 0){
            addSTF<<<nsrc, ntask2>>>(
                dat::dsx, dat::dsy, dat::dsz, dat::stf_x, dat::stf_y, dat::stf_z,
                dat::src_x_id, dat::src_z_id, dat::isrc[0], sh, psv, it, nx, 0
            );
        }
        else if(mode == 1){
            addSTF<<<nrec, ntask2>>>(
                dat::dsx, dat::dsy, dat::dsz, dat::adstf_x, dat::adstf_y, dat::adstf_z,
                dat::rec_x_id, dat::rec_z_id, -1, sh, psv, it, nx, nt
            );
        }
        if(sh){
            updateV<<<nxb2, nzt>>>(dat::vy, dat::dsy, dat::rho, dat::absbound, dt);
            divVY<<<nxb2, nzt>>>(dat::dvydx, dat::dvydz, dat::vy, dat::X, dat::Z, nx, nz);
            updateSY<<<nxb2, nzt>>>(dat::sxy, dat::szy, dat::dvydx, dat::dvydz, dat::mu, dt);
            updateU<<<nxb2, nzt>>>(dat::uy, dat::vy, dt);
        }
        if(psv){
            updateV<<<nxb2, nzt>>>(dat::vx, dat::dsx, dat::rho, dat::absbound, dt);
            updateV<<<nxb2, nzt>>>(dat::vz, dat::dsz, dat::rho, dat::absbound, dt);
            divVXZ<<<nxb2, nzt>>>(dat::dvxdx, dat::dvxdz, dat::dvzdx, dat::dvzdz, dat::vx, dat::vz, dat::X, dat::Z, nx, nz);
            updateSXZ<<<nxb2, nzt>>>(dat::sxx, dat::szz, dat::sxz, dat::dvxdx, dat::dvxdz, dat::dvzdx, dat::dvzdz, dat::lambda, dat::mu, dt);
            updateU<<<nxb2, nzt>>>(dat::ux, dat::vx, dt);
            updateU<<<nxb2, nzt>>>(dat::uz, dat::vz, dt);
        }
        if(mode == 0){
            if(dat::obs_type == 0){
                saveRec<<<nrec, ntask2>>>(
                    dat::out_x, dat::out_y, dat::out_z, dat::vx, dat::vy, dat::vz,
                    dat::rec_x_id, dat::rec_z_id, nx, nt, sh, psv, it
                );
            }
            else if(dat::obs_type == 1){
                saveRec<<<nrec, ntask2>>>(
                    dat::out_x, dat::out_y, dat::out_z, dat::ux, dat::uy, dat::uz,
                    dat::rec_x_id, dat::rec_z_id, nx, nt, sh, psv, it
                );
            }
            else if(dat::obs_type == 2 && dat::isrc[0] >= 0){
                saveRec<<<nrec, ntask2>>>(
                    dat::u_obs_x, dat::u_obs_y, dat::u_obs_z, dat::ux, dat::uy, dat::uz,
                    dat::rec_x_id, dat::rec_z_id, dat::isrc[0], nx, sh, psv, it
                );
            }
            if((it + 1) % dat::sfe == 0){
                int isfe = dat::nsfe - (it + 1) / dat::sfe;
                if(sh){
                    mat::copyDeviceToHost(dat::vy_forward[isfe], dat::vy, nx2, nz);
                }
                if(psv){
                    mat::copyDeviceToHost(dat::vx_forward[isfe], dat::vx, nx2, nz);
                    mat::copyDeviceToHost(dat::vz_forward[isfe], dat::vz, nx2, nz);
                }
            }
        }
        else if(mode == 1){
            if((it + dat::sfe) % dat::sfe == 0){
                // dsi -> ui_fw -> vi_fw
                int isfe = (it + dat::sfe) / dat::sfe - 1;
                float tsfe = dat::sfe * dt;
                if(sh){
                    mat::copyHostToDevice(dat::dsy, dat::uy_forward[isfe], nx2, nz);
                    divVY<<<nxb2, nzt>>>(dat::dvydx, dat::dvydz, dat::uy, dat::X, dat::Z, nx, nz);
                    divVY<<<nxb2, nzt>>>(dat::dvydx_fw, dat::dvydz_fw, dat::dsy, dat::X, dat::Z, nx, nz);
                    mat::copyHostToDevice(dat::dsy, dat::vy_forward[isfe], nx2, nz);
                    interactionRhoY<<<nxb2, nzt>>>(dat::K_rho, dat::vy, dat::dsy, tsfe);
                    interactionMuY<<<nxb2, nzt>>>(dat::K_mu, dat::dvydx, dat::dvydx_fw, dat::dvydz, dat::dvydz_fw, tsfe);
                }
                if(psv){
                    mat::copyHostToDevice(dat::dsx, dat::ux_forward[isfe], nx2, nz);
                    mat::copyHostToDevice(dat::dsz, dat::uz_forward[isfe], nx2, nz);
                    divVXZ<<<nxb2, nzt>>>(
                        dat::dvxdx, dat::dvxdz, dat::dvzdx, dat::dvzdz,
                        dat::ux, dat::uz, dat::X, dat::Z, nx, nz
                    );
                    divVXZ<<<nxb2, nzt>>>(
                        dat::dvxdx_fw, dat::dvxdz_fw, dat::dvzdx_fw, dat::dvzdz_fw,
                        dat::dsx, dat::dsz, dat::X, dat::Z, nx, nz
                    );

                    mat::copyHostToDevice(dat::dsx, dat::vx_forward[isfe], nx2, nz);
                    mat::copyHostToDevice(dat::dsz, dat::vz_forward[isfe], nx2, nz);
                    interactionRhoXZ<<<nxb2, nzt>>>(dat::K_rho, dat::vx, dat::dsx, dat::vz, dat::dsz, tsfe);
                    interactionMuXZ<<<nxb2, nzt>>>(
                        dat::K_mu, dat::dvxdx, dat::dvxdx_fw, dat::dvxdz, dat::dvxdz_fw,
                        dat::dvzdx, dat::dvzdx_fw, dat::dvzdz, dat::dvzdz_fw, tsfe
                    );
                    interactionLambdaXZ<<<nxb2, nzt>>>(dat::K_lambda, dat::dvxdx, dat::dvxdx_fw, dat::dvzdz, dat::dvzdz_fw, tsfe);
                }
            }
        }
    }
}
static void runForward(int isrc0, int isrc1){
    dat::simulation_mode = 0;
    dat::isrc[0] = isrc0;
    dat::isrc[1] = isrc1;
    runWaveFieldPropagation();
}
static void runAdjoint(int init_kernel){
    dat::simulation_mode = 1;
    if(init_kernel){
        initialiseKernels();
    }
    runWaveFieldPropagation();
}
static void initialiseFilters(){
    // taper weights
    dat::tw = mat::create(nt);
    getTaperWeights<<<nt, 1>>>(dat::tw, dt, nt);

    // gaussian filter
    if(dat::filter_kernel){
        dat::gsum = mat::create(nx, nz);
        dat::gtemp = mat::create(nx, nz);
        initialiseGaussian<<<nxb, nzt>>>(dat::gsum, nx, nz, dat::filter_kernel);
    }
}
static void prepareObs(){
    dat::obs_type = 2;
    prepareSTF();
    if(dat::obs_su){
        printf("Reading observed data");
        readSU();
    }
    else{
        printf("Generating observed data\n");
        loadModel(dat::model_true);
        for(int isrc = 0; isrc < nsrc; isrc += ntask){
            runForward(isrc, getTaskIndex(isrc));
            printStat(isrc, nsrc);
        }
    }
    initialiseFilters();
    dat::obs_type = 1;
}
static float calculateEnvelopeMisfit(float **adstf, float *d_misfit, float **out, float ***u_obs,
    cufftComplex *syn, cufftComplex *obs, float *esyn, float *eobs, float *ersd, float *etmp,
    float dt, int isrc, int irec, int j, int nt){
    copyWaveform<<<nt, 1>>>(d_misfit, u_obs, isrc+j, irec);
    hilbert(d_misfit, obs);
    copyWaveform<<<nt, 1>>>(d_misfit, out, irec, j*nt);
    hilbert(d_misfit, syn);

    copyC2Abs<<<nt, 1>>>(esyn, syn, nt);
    copyC2Abs<<<nt, 1>>>(eobs, obs, nt);
    float max = mat::amax(esyn, nt) * 0.05;
    if(max < 0.05){
        max = 0.05;
    }
    envelopetmp<<<nt, 1>>>(etmp, esyn, eobs, max);

    copyC2Imag<<<nt, 1>>>(ersd, syn, nt);
    mat::calc(ersd, ersd, etmp, nt);
    hilbert(ersd, obs);
    copyC2Imag<<<nt, 1>>>(ersd, obs, nt);

    prepareEnvelopeSTF<<<nt, 1>>>(adstf, etmp, d_misfit, ersd, nt, irec, j*nt);
    mat::calc(ersd, 1, esyn, -1, eobs, nt);
    return mat::norm(ersd, nt) * sqrt(dt);
}
static float computeKernelsAndMisfit(int kernel){
    float misfit = 0;
    float *d_misfit = mat::create(nt);
    cufftComplex *syn;
    cufftComplex *obs;
    float *esyn;
    float *eobs;
    float *ersd;
    float *etmp;
    if(dat::misfit_type == 1){
        cudaMalloc((void**)&syn, nt * sizeof(cufftComplex));
        cudaMalloc((void**)&obs, nt * sizeof(cufftComplex));
        esyn = mat::create(nt);
        eobs = mat::create(nt);
        ersd = mat::create(nt);
        etmp = mat::create(nt);
    }

    if(kernel){
        printf("Computing gradient\n");
        initialiseKernels();
    }
    if(!sh){
        mat::init(dat::adstf_y, 0, nrec, nt * ntask);
    }
    if(!psv){
        mat::init(dat::adstf_x, 0, nrec, nt * ntask);
        mat::init(dat::adstf_z, 0, nrec, nt * ntask);
    }
    for(int isrc = 0; isrc < nsrc; isrc+=ntask){
        int jsrc = getTaskIndex(isrc);
        runForward(isrc, jsrc);
        for(int j = 0; j <= jsrc - isrc; j++){
            for(int irec = 0; irec < nrec; irec++){
                if(dat::misfit_type == 1){
                    if(sh){
                        misfit += calculateEnvelopeMisfit(dat::adstf_y, d_misfit, dat::out_y, dat::u_obs_y,
                            syn, obs, esyn, eobs, ersd, etmp, dt, isrc, irec, j, nt);
                    }
                    if(psv){
                        misfit += calculateEnvelopeMisfit(dat::adstf_x, d_misfit, dat::out_x, dat::u_obs_x,
                            syn, obs, esyn, eobs, ersd, etmp, dt, isrc, irec, j, nt);
                        misfit += calculateEnvelopeMisfit(dat::adstf_z, d_misfit, dat::out_z, dat::u_obs_z,
                            syn, obs, esyn, eobs, ersd, etmp, dt, isrc, irec, j, nt);
                    }
                }
                else{
                    if(sh){
                        calculateWaveformMisfit<<<nt, 1>>>(dat::adstf_y, d_misfit, dat::out_y, dat::u_obs_y, dat::tw, isrc, irec, j, nt);
                        misfit += mat::norm(d_misfit, nt) * sqrt(dt);
                    }
                    if(psv){
                        calculateWaveformMisfit<<<nt, 1>>>(dat::adstf_x, d_misfit, dat::out_x, dat::u_obs_x, dat::tw, isrc, irec, j, nt);
                        misfit += mat::norm(d_misfit, nt) * sqrt(dt);
                        calculateWaveformMisfit<<<nt, 1>>>(dat::adstf_z, d_misfit, dat::out_z, dat::u_obs_z, dat::tw, isrc, irec, j, nt);
                        misfit += mat::norm(d_misfit, nt) * sqrt(dt);
                    }
                }
            }
        }

        if(kernel){
            runAdjoint(0);
            printStat(isrc, nsrc);
        }
    }

    cudaFree(d_misfit);
    if(dat::misfit_type == 1){
        cudaFree(syn);
        cudaFree(obs);
        cudaFree(esyn);
        cudaFree(eobs);
        cudaFree(ersd);
        cudaFree(etmp);
    }

    if(kernel){
        if(dat::filter_kernel){
            filterKernelX<<<nxb, nzt>>>(dat::K_rho, dat::gtemp, nx, dat::filter_kernel);
            filterKernelZ<<<nxb, nzt>>>(dat::K_rho, dat::gtemp, dat::gsum, nz, dat::filter_kernel);
            filterKernelX<<<nxb, nzt>>>(dat::K_mu, dat::gtemp, nx, dat::filter_kernel);
            filterKernelZ<<<nxb, nzt>>>(dat::K_mu, dat::gtemp, dat::gsum, nz, dat::filter_kernel);
            filterKernelX<<<nxb, nzt>>>(dat::K_lambda, dat::gtemp, nx, dat::filter_kernel);
            filterKernelZ<<<nxb, nzt>>>(dat::K_lambda, dat::gtemp, dat::gsum, nz, dat::filter_kernel);
        }
    }

    return misfit;
}
static float calculateMisfit(){
    return computeKernelsAndMisfit(0);
}
static float computeKernels(){
    return computeKernelsAndMisfit(1);
}
static int computeDirectionCG(float **p_new, float **p_old, float **g_new, float **g_old){
    dat::inv_count++;
    if(dat::inv_count == 1){
        mat::copy(p_new, g_new, -1, nx, nz);
        return 0;
    }
    else if(dat::inv_maxiter && dat::inv_count > dat::inv_maxiter){
        fprintf(dat::logfile, "  restarting NLCG... [periodic restart]\n");
        printf("  restarting NLCG... [periodic restart]\n");
        return -1;
    }
    // self.precond: later

    float den = mat::dot(g_old, g_old, nx, nz);
    mat::calc(p_new, 1, g_new, -1, g_old, nx, nz);
    float num = mat::dot(g_new, p_new, nx, nz);
    float beta = num / den;
    mat::calc(p_new, -1, g_new, beta, p_old, nx, nz);

    // lose of conjugacy? later
    if(mat::dot(p_new, g_new, nx, nz) > 0){
        fprintf(dat::logfile, "  restarting NLCG... [not a descent direction]\n");
        printf("  restarting NLCG... [not a descent direction]\n");
        return -1;
    }
    return 1;
}
static int computeDirectionLBFGS(float **p_new, float **p_old, float **g_new, float **g_old, float **m_new, float **m_old){
    dat::inv_count++;
    if(dat::inv_count == 1){
        mat::copy(p_new, g_new, -1, nx, nz);
        return 0;
    }
    else if(dat::inv_maxiter && dat::inv_count > dat::inv_maxiter){
        fprintf(dat::logfile, "  restarting LBFGS... [periodic restart]\n");
        printf("  restarting LBFGS... [periodic restart]\n");
        return -1;
    }

    float **tmpS = dat::lbfgs_S[dat::lbfgs_mem-1];
    float **tmpY = dat::lbfgs_Y[dat::lbfgs_mem-1];
    for(int i = dat::lbfgs_mem-1; i > 0; i--){
        dat::lbfgs_S[i] = dat::lbfgs_S[i-1];
        dat::lbfgs_Y[i] = dat::lbfgs_Y[i-1];
    }
    dat::lbfgs_S[0] = tmpS;
    dat::lbfgs_Y[0] = tmpY;

    mat::calc(p_old, 1, m_new, -1, m_old, nx, nz);
    mat::copyDeviceToHost(dat::lbfgs_S[0], p_old, nx, nz);
    mat::calc(p_old, 1, g_new, -1, g_old, nx, nz);
    mat::copyDeviceToHost(dat::lbfgs_Y[0], p_old, nx, nz);

    if(dat::lbfgs_used < dat::lbfgs_mem){
        dat::lbfgs_used++;
    }

    int &kk = dat::lbfgs_used;
    float *rh = mat::createHost(kk);
    float *al = mat::createHost(kk);

    // S->m_old  Y->p_old
    mat::copy(p_new, g_new, nx, nz);
    float sty, yty;
    for(int i = 0; i < kk; i++){
        mat::copyHostToDevice(m_old, dat::lbfgs_S[i], nx, nz);
        mat::copyHostToDevice(p_old, dat::lbfgs_Y[i], nx, nz);
        rh[i] = 1 / mat::dot(p_old, m_old, nx, nz);
        al[i] = rh[i] * mat::dot(m_old, p_new, nx, nz);
        mat::calc(p_new, 1, p_new, -al[i], p_old, nx, nz);
        if(i == 0){
            sty = 1 / rh[i];
            yty = mat::dot(p_old, p_old, nx, nz);
        }
    }
    mat::copy(p_new, p_new, sty/yty, nx, nz);

    for(int i = kk-1; i >= 0; i--){
        mat::copyHostToDevice(m_old, dat::lbfgs_S[i], nx, nz);
        mat::copyHostToDevice(p_old, dat::lbfgs_Y[i], nx, nz);
        float be = rh[i] * mat::dot(p_old, p_new, nx, nz);
        mat::calc(p_new, 1, p_new, al[i] - be, m_old, nx, nz);
    }

    free(rh);
    free(al);

    float angle = calculateAngle(p_new, g_new, 1, nx, nz);
    if(angle>=pi/2 || angle<=0){
        fprintf(dat::logfile, "  restarting LBFGS... [not a descent direction]\n");
        printf("  restarting LBFGS... [not a descent direction]\n");
        return -1;
    }
    mat::copy(p_new, p_new, -1, nx, nz);

    return 1;
}
static int argmin(float *f, int n){
    float min = f[0];
    int idx = 0;
    for(int i = 1; i < n; i++){
        if(f[i] < min){
            min = f[i];
            idx = i;
        }
    }
    return idx;
}
static int checkBracket(float *x, float *f, int n){
    int imin = argmin(f, n);
    float fmin = f[imin];
    if(fmin < f[0]){
        for(int i = imin; i < n; i++){
            if(f[i] > fmin){
                return 1;
            }
        }
    }
    return 0;
}
static int goodEnough(float *x, float *f, int n, float *alpha){
    float thresh = log10(dat::ls_thresh);
    if(!checkBracket(x, f, n)){
        return 0;
    }
    float p[3];
    int idx = argmin(f, n) - 1;
    int fitlen;
    if(idx + 3 >= n){
        fitlen = 3;
    }
    else{
        fitlen = 4;
    }
    polyfit(x + idx, f + idx, p, fitlen);
    if(p[0] <= 0){
        printf("line search error\n");
    }
    else{
        float x0 = -p[1]/(2*p[0]);
        *alpha = x0;
        for(int i = 1; i < n; i++){
            if(fabs(log10(x[i]/x0)) < thresh){
                return 1;
            }
        }
    }
    return 0;
}
static float backtrack2(float f0, float g0, float x1, float f1, float b1, float b2){
    float x2 = -g0 * x1 * x1 / (2  *(f1 - f0 - g0 * x1));
    if(x2 > b2*x1){
        x2 = b2*x1;
    }
    else if(x2 < b1*x1){
        x2 = b1*x1;
    }
    return x2;
}
static float updateModel(float **m, float **p, float alpha, float alpha_old){
    updateModel<<<nxb, nzt>>>(m, p, alpha - alpha_old);
    return alpha;
}
static float calculateStep(const int step_count, float step_len_max, int *status){
    float update_count = -1;
    float alpha;

    float *x = mat::createHost(step_count+1);
    float *f = mat::createHost(step_count+1);
    for(int i = 0; i < step_count+1; i++){
        int j = dat::ls_count - 1 - step_count + i;
        x[i] = dat::step_lens[j];
        f[i] = dat::func_vals[j];
    }
    for(int i = 0; i < step_count+1; i++){
        for(int j = i+1; j < step_count+1; j++){
            if(x[j] < x[i]){
                float tmp;
                tmp = x[i]; x[i] = x[j]; x[j] = tmp;
                tmp = f[i]; f[i] = f[j]; f[j] = tmp;
            }
        }
    }
    for(int i = 0; i < dat::ls_count; i++){
        if(fabs(dat::step_lens[i]) < 1e-6){
            update_count++;
        }
    }
    if(step_count == 0){
        if(update_count == 0){
            alpha = 1 / dat::ls_gtg[dat::inv_count - 1];
            *status = 0;
        }
        else{
            int idx = argmin(dat::func_vals, dat::ls_count - 1);
            alpha = dat::step_lens[idx] * dat::ls_gtp[dat::inv_count - 2] / dat::ls_gtp[dat::inv_count - 1];
            *status = 0;
        }
    }
    else if(checkBracket(x, f, step_count+1)){
        if(goodEnough(x, f, step_count+1, &alpha)){
            alpha = x[argmin(f, step_count+1)];
            *status = 1;
        }
        else{
            *status = 0;
        }
    }
    else if(step_count <= dat::ls_stepcountmax){
        int i;
        for(i = 1; i < step_count+1; i++){
            if(f[i] > f[0]) break;
        }

        if(i == step_count+1){
            alpha = 1.618034 * x[step_count];
            *status = 0;
        }
        else{
            float slope = dat::ls_gtp[dat::inv_count-1]/dat::ls_gtg[dat::inv_count-1];
            alpha = backtrack2(f[0], slope, x[1], f[1], 0.1, 0.5);
            *status = 0;
        }
    }
    else{
        alpha = 0;
        *status = -1;
    }

    if(alpha > step_len_max){
        if(step_count == 0){
            alpha = 0.618034 * step_len_max;
            *status = 0;
        }
        else{
            alpha = step_len_max;
            *status = 1;
        }
    }

    free(x);
    free(f);

    return alpha;
}
static float calculateStepBT(const int step_count, float step_len_max, int *status){
    float update_count = -1;
    for(int i = 0; i < dat::ls_count; i++){
        if(fabs(dat::step_lens[i]) < 1e-6){
            update_count++;
        }
    }
    if(update_count == 0){
        return calculateStep(step_count, step_len_max, status);
    }

    float alpha;

    float *x = mat::createHost(step_count+1);
    float *f = mat::createHost(step_count+1);
    for(int i = 0; i < step_count+1; i++){
        int j = dat::ls_count - 1 - step_count + i;
        x[i] = dat::step_lens[j];
        f[i] = dat::func_vals[j];
    }
    for(int i = 0; i < step_count+1; i++){
        for(int j = i+1; j < step_count+1; j++){
            if(x[j] < x[i]){
                float tmp;
                tmp = x[i]; x[i] = x[j]; x[j] = tmp;
                tmp = f[i]; f[i] = f[j]; f[j] = tmp;
            }
        }
    }

    int idx = argmin(f, step_count+1);
    if(step_count == 0){
        alpha = step_len_max;
        if(alpha > 1){
            alpha = 1;
        }
        *status = 0;
    }
    else if(f[idx] < f[0]){
        alpha = x[idx];
        *status = 1;
    }
    else if(step_count <= dat::ls_stepcountmax){
        float slope = dat::ls_gtp[dat::inv_count-1]/dat::ls_gtg[dat::inv_count-1];
        alpha = backtrack2(f[0], slope, x[1], f[1], 0.1, 0.5);
        *status = 0;
    }
    else{
        alpha = 0;
        *status = -1;
    }

    free(x);
    free(f);

    return alpha;
}
static void restartSearch(float **p, float **g){
    mat::copy(p, g, -1, nx, nz);
    dat::ls_count = 0;
    dat::inv_count = 1;
    if(dat::optimize == 1){
        dat::lbfgs_used = 0;
    }
}
static void lineSearch(float **m, float **g, float **p, float f){
    if(dat::display_mode != 0){
        printf("\n");
    }
    printf("Performing line search\n");
    int status = 0;
    float alpha = 0;

    float norm_m = mat::amax(m, nx, nz);
    float norm_p = mat::amax(p, nx, nz);
    float gtg = mat::dot(g, g, nx, nz);
    float gtp = mat::dot(g, p, nx, nz);

    float step_len_max = dat::ls_steplenmax * norm_m / norm_p;
    int step_count = 0;
    dat::step_lens[dat::ls_count] = 0;
    dat::func_vals[dat::ls_count] = f;
    dat::ls_gtg[dat::inv_count-1] = gtg;
    dat::ls_gtp[dat::inv_count-1] = gtp;
    dat::ls_count++;

    float alpha_old = 0;

    if(dat::ls_stepleninit && dat::ls_count <= 1){
        alpha = dat::ls_stepleninit * norm_m / norm_p;
    }
    else{
        alpha = calculateStep(step_count, step_len_max, &status);
    }

    while(1){
        alpha_old = updateModel(m, p, alpha, alpha_old);
        dat::step_lens[dat::ls_count] = alpha;
        dat::func_vals[dat::ls_count] = calculateMisfit();
        dat::ls_count++;
        step_count++;
        dat::neval++;

        if(dat::optimize == 1){
            alpha = calculateStepBT(step_count, step_len_max, &status);
        }
        else{
            alpha = calculateStep(step_count, step_len_max, &status);
        }
        if(step_count < 10){
            fprintf(dat::logfile, "  step 0%d  misfit = %f\n", step_count, dat::func_vals[dat::ls_count-1]/dat::misfit_ref);
            printf("  step 0%d  misfit = %f\n", step_count, dat::func_vals[dat::ls_count-1]/dat::misfit_ref);
        }
        else{
            fprintf(dat::logfile, "  step %d  misfit = %f\n", step_count, dat::func_vals[dat::ls_count-1]/dat::misfit_ref);
            printf("  step %d  misfit = %f\n", step_count, dat::func_vals[dat::ls_count-1]/dat::misfit_ref);
        }
        if(status > 0){
            fprintf(dat::logfile, "  alpha = %.2e\n", alpha);
            printf("  alpha = %.2e\n", alpha);
            float angle =  calculateAngle(p, g, -1, nx, nz)*180/pi;
            fprintf(dat::logfile, "  angle = %f\n\n", angle);
            printf("  angle = %f\n", angle);
            updateModel(m, p, alpha, alpha_old);
            for(int i = 0; ; i++){
                if(dat::opti_history[i][0] < 0){
                    dat::opti_history[i][0] = (float)dat::neval;
                    dat::opti_history[i][1] = dat::func_vals[argmin(dat::func_vals, dat::ls_count)]/dat::misfit_ref;
                    break;
                }
            }
            return;
        }
        else if(status < 0){
            updateModel(m, p, 0, alpha_old);
            if(calculateAngle(p, g, -1, nx, nz) < 1e-3){
                printf("  line search failed\n");
                dat::inv_iteration = 0;
                return;
            }
            else{
                printf("  restarting line search...\n");
                restartSearch(p, g);
                lineSearch(m, g, p, f);
            }
        }
    }
}
static void inversionRoutine(){
    cublasCreate(&cublas_handle);
    cusolverDnCreate(&solver_handle);
    if(dat::misfit_type == 1){
        cufftPlan1d(&cufft_handle, nt, CUFFT_C2C, 1);
    }
    if(dat::optimize == 1){
        dat::lbfgs_used = 0;
    }

    {
        char parbuffer[81];
        FILE *parfile = fopen(dat::parfile, "r");
        sprintf(parbuffer, "%s/cfg", dat::output_path);
        FILE *outfile = fopen(parbuffer, "w");
        sprintf(parbuffer, "%s/src", dat::output_path);
        FILE *srcfile = fopen(parbuffer, "w");
        sprintf(parbuffer, "%s/rec", dat::output_path);
        FILE *recfile = fopen(parbuffer, "w");
        sprintf(parbuffer, "%s/log", dat::output_path);
        dat::logfile = fopen(parbuffer,"w");
        dat::neval = 0;

        fprintf(outfile, "%s\n", dat::argstr);
        while(fgets(parbuffer, 81, parfile) != NULL){
            if(parbuffer[0] == '#'){
                continue;
            }
            for(int i = 0; i < 81 && parbuffer[i] != '\0'; i++){
                if(parbuffer[i] == '#'){
                    parbuffer[i] = '\n';
                    parbuffer[i+1] = '\0';
                    break;
                }
            }
            fprintf(outfile, "%s", parbuffer);
        }

        float *src_x = mat::createHost(nsrc);
        float *src_z = mat::createHost(nsrc);
        mat::copyDeviceToHost(src_x, dat::src_x, nsrc);
        mat::copyDeviceToHost(src_z, dat::src_z, nsrc);
        for(int i = 0; i < nsrc; i++){
            fprintf(srcfile, "%.5e %.5e\n", src_x[i], src_z[i]);
        }

        float *rec_x = mat::createHost(nrec);
        float *rec_z = mat::createHost(nrec);
        mat::copyDeviceToHost(rec_x, dat::rec_x, nrec);
        mat::copyDeviceToHost(rec_z, dat::rec_z, nrec);
        for(int i = 0; i < nrec; i++){
            fprintf(recfile, "%.5e %.5e\n", rec_x[i], rec_z[i]);
        }

        free(src_x);
        free(src_z);
        free(rec_x);
        free(rec_z);

        fclose(parfile);
        fclose(srcfile);
        fclose(recfile);
        fclose(outfile);
    }

    prepareObs();
    exportData(-1);
    loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);

    float **m_new;
    float **m_old;
    float **g_new;
    switch(dat::inv_parameter){
        case 0: m_new = dat::lambda; g_new = dat::K_lambda; break;
        case 1: m_new = dat::mu; g_new = dat::K_mu; break;
        case 2: m_new = dat::rho; g_new = dat::K_rho; break;
    }
    if(dat::optimize == 1){
        dat::lbfgs_S = mat::createHost(dat::lbfgs_mem, nx, nz);
        dat::lbfgs_Y = mat::createHost(dat::lbfgs_mem, nx, nz);
        m_old = mat::create(nx, nz);
    }

    float **g_old = mat::create(nx, nz);
    float **p_old = mat::create(nx, nz);
    float **p_new = mat::create(nx, nz);

    int opti_max = dat::inv_iteration * dat::ls_stepcountmax;
    dat::func_vals = mat::createHost(opti_max);
    dat::step_lens = mat::createHost(opti_max);
    dat::opti_history = mat::createHost(opti_max, 2);
    mat::initHost(dat::opti_history, -1, opti_max, 2);

    dat::ls_gtg = mat::createHost(dat::inv_iteration);
    dat::ls_gtp = mat::createHost(dat::inv_iteration);
    dat::ls_count = 0;
    dat::inv_count = 0;

    clock_t timestart = clock();
    for(int iter = 0; iter < dat::inv_iteration; iter++){
        fprintf(dat::logfile, "iteration %d / %d\n", iter + 1, dat::inv_iteration);
        if(dat::display_mode != 0){
            printf("\n");
        }
        printf("\nStarting iteration %d / %d\n", iter + 1, dat::inv_iteration);
        float f = computeKernels();
        if(iter == 0){
            dat::misfit_ref = f;
        }
        dat::neval += 2;

        int dir;
        if(dat::optimize == 0){
            dir = computeDirectionCG(p_new, p_old, g_new, g_old);
        }
        else{
            dir = computeDirectionLBFGS(p_new, p_old, g_new, g_old, m_new, m_old);
            mat::copy(m_old, m_new, nx, nz);
        }
        if(dir < 0){
            restartSearch(p_new, g_new);
        }
        lineSearch(m_new, g_new, p_new, f);

        mat::copy(p_old, p_new, nx, nz);
        mat::copy(g_old, g_new, nx, nz);
        exportData(iter);
    }
    int unsharp = lroundf(dat::unsharp_mask * dat::filter_kernel);
    if(unsharp > 0){
        applyUnsharpMask(dat::lambda, unsharp, 2);
        applyUnsharpMask(dat::mu, unsharp, 2);
        applyUnsharpMask(dat::rho, unsharp, 2);
        exportData(dat::inv_iteration);
    }

    fprintf(dat::logfile, "misfit\n  %f\n", dat::misfit_ref);
    for(int i = 0; lroundf(dat::opti_history[i][0]) >=0; i++){
        fprintf(dat::logfile, "  %ld %f\n", lroundf(dat::opti_history[i][0]), dat::opti_history[i][1]);
    }

    float et = (float)(clock() - timestart) / CLOCKS_PER_SEC;
    if(et > 60){
        int etmin = (int)(et / 60);
        fprintf(dat::logfile, "\nelapsed time: %dmin %lds\n", etmin, lroundf(et - etmin*60));
    }
    else{
        fprintf(dat::logfile, "\nelapsed time: %.2fs\n", et);
    }

    fclose(dat::logfile);
    cublasDestroy(cublas_handle);
    cusolverDnDestroy(solver_handle);
    if(dat::misfit_type == 1){
        cufftDestroy(cufft_handle);
    }
}

int main(int argc, const char *argv[]){
    char argstr[6000] = {'\0'};
    dat::argstr = argstr;
    dat::parfile = copyString("config");
    dat::output_path = copyString("output");
    dat::model_init = copyString("model_init");
    dat::model_true = copyString("model_true");
    dat::model_p = -1;
    dat::model_s = -1;
    dat::model_r = -1;
    dat::nsrc = 1;
    dat::nrec = 1;
    dat::inv_iteration = 5;
    dat::simulation_mode = 0;

    if(argc > 1){
        for(int i = 1; i < argc; i += 2){
            if(strcmp(argv[i], "-c") == 0){
                dat::parfile = copyString(argv[i+1]);
            }
        }
    }

    clock_t timestart = clock();
    if(importData(dat::parfile, argc, argv)){
        switch(mode){
            case 0:{
                inversionRoutine();
                break;
            }
            case 1:{
                loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);
                prepareSTF();
                dat::ntask = 1;
                runForward(-1, -1);
                char buffer[80];
                char itext[10];
                for(int i = 0; i < dat::nsfe; i++){
                    if(i+1 < 10){
                        sprintf(itext, "000%d", i+1);
                    }
                    else if(i+1 < 100){
                        sprintf(itext, "00%d", i+1);
                    }
                    else if(i+1 < 1000){
                        sprintf(itext, "0%d", i+1);
                    }
                    else{
                        sprintf(itext, "%d", i+1);
                    }
                    if(sh){
                        sprintf(buffer, "%s/vy_%.2f_%s.bin", dat::output_path, dat::sfe*dt, itext);
                        mat::write(dat::vy_forward[i], nx, nz, buffer);
                    }
                    if(psv){
                        sprintf(buffer, "%s/vx_%.2f_%s.bin", dat::output_path, dat::sfe*dt, itext);
                        mat::write(dat::vx_forward[i], nx, nz, buffer);
                        sprintf(buffer, "%s/vz_%.2f_%s.bin", dat::output_path, dat::sfe*dt, itext);
                        mat::write(dat::vz_forward[i], nx, nz, buffer);
                    }

                }
                writeSU();
                break;
            }
            case 2:{
                cublasCreate(&cublas_handle);
                if(dat::misfit_type == 1){
                    cufftPlan1d(&cufft_handle, nt, CUFFT_C2C, 1);
                }
                prepareObs();
                if(dat::obs_su){
                    printf("\n");
                }
                printf("\n");
                loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);
                computeKernels();
                exportData(0);
                cublasDestroy(cublas_handle);
                if(dat::misfit_type == 1){
                    cufftDestroy(cufft_handle);
                }
                break;
            }
            case 10:{
                dat::obs_su = 0;
                prepareObs();
                writeSU();
                break;
            }
            case 11:{
                loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);
                generateChecker(dat::mu, 0.1, 0.5, 2, 2);
                exportData(-1);
                break;
            }
            case 12:{
                loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);
                generateLayer(dat::mu, 0.1, 5);
                exportData(-1);
                break;
            }
            case 13:{
                loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);
                generateRandomLayer(dat::mu, 0.1, 0.4, 5);
                exportData(-1);
                break;
            }
            case 15:{
                loadModel(dat::model_init, dat::model_p, dat::model_s, dat::model_r);
                mat::copy(dat::mu, dat::mu, 0.64, nx, nz);
                exportData(-1);
                break;
            }
            case 16:{
                dat::output_path = copyString("output");
                loadModel("model/marmousi");
                // mat::copy(dat::mu, dat::mu, 0.64, nx, nz);
                exportData(-1);
                break;
            }
            case 17:{
                printf("hello workd\n");
            }
        }
    }
    else{
        printf("error loading data\n");
    }
    float et = (float)(clock() - timestart) / CLOCKS_PER_SEC;
    if(et > 60){
        int etmin = (int)(et / 60);
        printf("\nElapsed time: %dmin %lds\n", etmin, lroundf(et - etmin*60));
    }
    else{
        printf("\nElapsed time: %.2fs\n", et);
    }
    checkMemoryUsage();

    return 0;
}
