use std::process::Command;
use std::fs;
fn getExecutablePath() -> std::io::Result<std::path::PathBuf> {
    let path = std::env::current_exe()?;
    Ok(path)
}

fn pre_dll_has_exist()->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll", "cudart64_110.dll",
    "uv.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll","asmjit.dll", "zlibwapi.dll", "nvToolsExt64_1.dll", 
    "nvfuser_codegen.dll", "cudnn64_8.dll", "fbgemm.dll", "fbjni.dll", "libiomp5md.dll", "libiompstubs5md.dll", 
    "cublas64_11.dll", "cublasLt64_11.dll","cudnn64_8.dll","cufft64_10.dll","cufftw64_10.dll"];
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap();
    let torch_folder = exec_dir_path
    .join("bin")
    .join("torch_dll");

    if torch_folder.exists() {
        fs::remove_dir_all(torch_folder);
    }
    for name in global_file_name.iter(){
        let newpath = exec_dir_path.join("bin").join(".").join(name);
        if !newpath.exists() {
            return false;
        }
    }
    return true;
}
fn main() {
    //get current cmd args
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().join("bin");
    while !pre_dll_has_exist() {
        // 创建一个新的 Command 结构体，并指定要执行的程序名称
        let mut command = Command::new(exec_dir_path.join("diverseupdate.exe")).current_dir(exec_dir_path.as_path()).spawn().unwrap();
        // 启动进程
        let output = command.wait_with_output().unwrap();

        // 打印输出结果
        println!("{}", String::from_utf8_lossy(&output.stdout));
        // println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    }
    // if !pre_dll_has_exist() {
    //     println!("The prerequisite  package don't download, it need to download first");
    //     return;
    // }

    let  mut cmd = Command::new(exec_dir_path.join("diverseshot.exe"));
    if args.len() >= 2 {
        //get project_name from args 0
        let arg = format!("--open_project={}", args.get(1).unwrap());
        cmd.arg(arg);
    }
    
    let path = format!("{};", exec_dir_path.join("sfm").to_string_lossy());
    println!("path: {}", path);
    cmd.env("PATH", path);
    cmd.current_dir(exec_dir_path.as_path());
    let child = cmd.spawn().unwrap();
    // child.detach();
    std::mem::forget(child);
    // println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
}
