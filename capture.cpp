#include "ninfer/ops/linear.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <map>

namespace ninfer::ops {
void linbak(const Tensor&, const Weight&, Tensor&, LinearPolicy, WorkspaceArena&, cudaStream_t);
void linbak(const Tensor&, const Weight&, Tensor&, cudaStream_t);

static void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void dump(const std::filesystem::path& path,const Tensor& t,cudaStream_t stream) {
    if(!t.is_contiguous()) throw std::runtime_error("Observer requires contiguous input");
    if(std::filesystem::exists(path)) throw std::runtime_error("Observer refuses overwrite");
    std::vector<char> data(t.bytes());
    check(cudaMemcpyAsync(data.data(),t.data,data.size(),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));
    std::ofstream file(path,std::ios::binary);file.write(data.data(),data.size());
    if(!file) throw std::runtime_error("Observer file write failed");
}
static void observe(const Tensor& x,const Weight& w,const Tensor& out,cudaStream_t stream) {
    const char* dir=std::getenv("NINFER_HEAD_CAPTURE_DIR");
    if(!dir||w.n!=248320||w.k!=5120||x.ne[1]!=1) return;
    cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone) return;
    const auto arm=std::filesystem::path(dir)/"armed.txt";
    std::ifstream input(arm);if(!input) return;
    std::string name;std::getline(input,name);input.close();
    if(name.empty()||name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")!=std::string::npos)
        throw std::runtime_error("Invalid observer capture name");
    const bool all=std::getenv("NINFER_HEAD_CAPTURE_ALL")!=nullptr;
    static std::map<std::string,int> counts;
    std::string label=name;
    if(all){char suffix[24];int index=counts[name]++;if(index>=256)return;
        std::snprintf(suffix,sizeof(suffix),"-%04d",index);label+=suffix;}
    dump(std::filesystem::path(dir)/(label+"-x.bf16"),x,stream);
    dump(std::filesystem::path(dir)/(label+"-logits.bf16"),out,stream);
    if(!all)std::filesystem::remove(arm);
    std::fprintf(stderr,"HEAD CAPTURE %s complete, original arithmetic\n",label.c_str());
}
void linear(const Tensor& x,const Weight& w,Tensor& out,LinearPolicy policy,WorkspaceArena& work,cudaStream_t stream) {
    linbak(x,w,out,policy,work,stream);observe(x,w,out,stream);
}
void linear(const Tensor& x,const Weight& w,Tensor& out,cudaStream_t stream) {
    linbak(x,w,out,stream);observe(x,w,out,stream);
}
}
