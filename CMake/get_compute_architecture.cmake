# Reference: https://stackoverflow.com/a/68223399
function(get_compute_architecture compute_architecture_var)
	include(FindCUDA/select_compute_arch)

	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
	string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

	set(${compute_architecture_var} ${CUDA_ARCH_LIST} PARENT_SCOPE)
endfunction()
