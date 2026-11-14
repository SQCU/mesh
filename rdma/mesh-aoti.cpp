/* design/compiler-native-assignment-2026-11-13.md; output_data/compiler-workgroup-20261113/executor/aoti-binding-contract.md */
/* torch 2.14 torch/csrc/inductor/aoti_package/model_package_loader.h AOTIModelPackageLoader, boxed_run */
/* torch 2.14 torch/csrc/inductor/aoti_torch/tensor_converter.h; aoti_torch/utils.h */
#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/ops/empty.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <infiniband/verbs.h>
#include <stdatomic.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-field-initializers"
extern "C" {
#include "mesh-collective.h"
}
#pragma clang diagnostic pop

using torch::inductor::AOTIModelPackageLoader;

static void report(const char *text, char *error, size_t room) {
    if (room) strlcpy(error, text, room);
}

static void release(AtenTensorHandle *given, size_t count) {
    std::vector<AtenTensorHandle> unstolen(given, given + count);
    torch::aot_inductor::free_unstolen_handles(unstolen);
    std::fill(given, given + count, nullptr);
}

extern "C" void *mesh_aoti_load(const char *package_path, const char *model_name, char *error, size_t room) {
    try {
        return new AOTIModelPackageLoader(package_path, model_name);
    } catch (const std::exception &failure) {
        report(failure.what(), error, room);
    } catch (...) {
        report("AOTIModelPackageLoader: a non-standard exception", error, room);
    }
    return nullptr;
}

/* torch 2.14 aoti_runtime/interface.h AOTInductorModelContainerRun */
static int ran(AOTIModelPackageLoader *loader, std::vector<at::Tensor> &&inputs, void **outputs, size_t output_count,
               char *error, size_t room) {
    std::vector<at::Tensor> results = loader->boxed_run(std::move(inputs));
    if (results.size() == output_count) {
        std::vector<AtenTensorHandle> made = torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(results);
        std::copy(made.begin(), made.end(), reinterpret_cast<AtenTensorHandle *>(outputs));
        return 0;
    }
    report(("boxed_run returned " + std::to_string(results.size()) + " outputs for " + std::to_string(output_count) +
            " output slots").c_str(), error, room);
    return 1;
}

extern "C" int mesh_aoti_run(void *model, void **inputs, size_t input_count, void **outputs, size_t output_count,
                             char *error, size_t room) {
    auto *given = reinterpret_cast<AtenTensorHandle *>(inputs);
    try {
        return ran(static_cast<AOTIModelPackageLoader *>(model),
                   torch::aot_inductor::alloc_tensors_by_stealing_from_handles(given, input_count), outputs, output_count,
                   error, room);
    } catch (const std::exception &failure) {
        report(failure.what(), error, room);
    } catch (...) {
        report("boxed_run: a non-standard exception", error, room);
    }
    release(given, input_count);
    return 1;
}

extern "C" void mesh_aoti_free(void *model) {
    delete static_cast<AOTIModelPackageLoader *>(model);
}

/* design/compiler-native-assignment-2026-11-13.md, prepared-input binding */
struct Binding {
    AOTIModelPackageLoader *loader;
    size_t input_count;
    std::vector<uint64_t> arguments, views, first;
    std::vector<at::Tensor> copies;
    int64_t position;
    at::Tensor invocation;
    uint64_t execution;

    std::vector<at::Tensor> bound(std::vector<at::Tensor> &&taken) {
        std::vector<at::Tensor> inputs(input_count);
        for (size_t k = 0; k < arguments.size(); k++)
            inputs[arguments[k]] = std::move(taken[k]);
        for (size_t v = 0; v < views.size(); v++)
            inputs[views[v]] = copies[first[v] + execution % (first[v + 1] - first[v])];
        if (position >= 0) {
            invocation.data_ptr<uint64_t>()[2] = execution;
            inputs[position] = invocation;
        }
        return inputs;
    }
};

extern "C" void *mesh_aoti_bind(void *model, size_t input_count, const uint64_t *arguments, size_t argument_count,
                                const uint64_t *views, const uint64_t *first, size_t view_count, void **copies,
                                int64_t invocation, uint64_t context, uint64_t plan, uint64_t execution, char *error,
                                size_t room) {
    auto *given = reinterpret_cast<AtenTensorHandle *>(copies);
    try {
        at::Tensor operand = at::empty({3}, at::TensorOptions().dtype(at::kUInt64));
        uint64_t *at = operand.data_ptr<uint64_t>();
        at[0] = context, at[1] = plan, at[2] = execution;
        return new Binding{static_cast<AOTIModelPackageLoader *>(model), input_count,
                           {arguments, arguments + argument_count}, {views, views + view_count},
                           {first, first + view_count + 1},
                           torch::aot_inductor::alloc_tensors_by_stealing_from_handles(given, first[view_count]),
                           invocation, operand, execution};
    } catch (const std::exception &failure) {
        report(failure.what(), error, room);
    } catch (...) {
        report("mesh_aoti_bind: a non-standard exception", error, room);
    }
    release(given, first[view_count]);
    return nullptr;
}

extern "C" int mesh_aoti_call(void *binding, void **arguments, size_t argument_count, void **outputs,
                              size_t output_count, char *error, size_t room) {
    auto *bound = static_cast<Binding *>(binding);
    auto *given = reinterpret_cast<AtenTensorHandle *>(arguments);
    try {
        int failed = ran(bound->loader,
                         bound->bound(torch::aot_inductor::alloc_tensors_by_stealing_from_handles(given, argument_count)),
                         outputs, output_count, error, room);
        bound->execution += !failed;
        return failed;
    } catch (const std::exception &failure) {
        report(failure.what(), error, room);
    } catch (...) {
        report("boxed_run: a non-standard exception", error, room);
    }
    release(given, argument_count);
    return 1;
}

extern "C" void mesh_aoti_unbind(void *binding) {
    delete static_cast<Binding *>(binding);
}

/* torch 2.14 _inductor/config.py aot_inductor.custom_ops_to_c_shims; _inductor/codegen/cpp_wrapper_cpu.py
   get_c_shim_func_name, generate_c_shim_fallback_kernel; _inductor/ir.py FallbackKernel.create */
struct Invocation {
    struct mesh_ctx *context;
    uint64_t plan, execution;
};

static Invocation invoked(AtenTensorHandle invocation) {
    const uint64_t *at = torch::aot_inductor::tensor_handle_to_tensor_pointer(invocation)->data_ptr<uint64_t>();
    return {reinterpret_cast<struct mesh_ctx *>(at[0]), at[1], at[2]};
}

static void check(const char *call, int status, int64_t transfer, uint64_t execution) {
    if (status)
        throw std::runtime_error(std::string(call) + " transfer " + std::to_string(transfer) + " execution " +
                                 std::to_string(execution) + ": " + strerror(status));
}

/* rdma/mesh-call.h mesh_host_started, mesh_transfers_start */
extern "C" AOTITorchError aoti_torch_cpu_mesh_start(AtenTensorHandle invocation, AtenTensorHandle *) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        Invocation at = invoked(invocation);
        mesh_host_started(at.context, at.plan, 0, at.execution);
        check("mesh_transfers_start", mesh_transfers_start(at.context, at.plan, at.execution + 1), -1, at.execution);
    });
}

/* rdma/mesh-collective.h mesh_host_publish */
extern "C" AOTITorchError aoti_torch_cpu_mesh_publish(AtenTensorHandle invocation, int64_t transfer, AtenTensorHandle,
                                                      AtenTensorHandle *) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
        Invocation at = invoked(invocation);
        mesh_host_publish(at.context, at.plan, static_cast<uint32_t>(transfer), at.execution);
    });
}

/* rdma/mesh-collective.h mesh_host_wait; design/compiler-native-assignment-2026-11-13.md, send completion before
   reuse */
static void waited(AtenTensorHandle invocation, int64_t transfer, int64_t delay, uint32_t direction) {
    Invocation at = invoked(invocation);
    uint64_t generation = at.execution - static_cast<uint64_t>(delay);
    if (at.execution >= static_cast<uint64_t>(delay))
        check("mesh_host_wait", mesh_host_wait(at.context, at.plan, static_cast<uint32_t>(transfer), generation,
                                               direction), transfer, generation);
}

/* rdma/mesh.h MESH_RECEIVE */
extern "C" AOTITorchError aoti_torch_cpu_mesh_wait(AtenTensorHandle invocation, int64_t transfer, int64_t delay,
                                                   AtenTensorHandle, AtenTensorHandle *) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ waited(invocation, transfer, delay, MESH_RECEIVE); });
}

/* rdma/mesh.h MESH_SEND */
extern "C" AOTITorchError aoti_torch_cpu_mesh_sent(AtenTensorHandle invocation, int64_t transfer, int64_t delay,
                                                   AtenTensorHandle, AtenTensorHandle *) {
    AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ waited(invocation, transfer, delay, MESH_SEND); });
}
