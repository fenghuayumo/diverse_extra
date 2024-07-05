use std::process::Command;

fn getExecutablePath() -> std::io::Result<std::path::PathBuf> {
    let path = std::env::current_exe()?;
    Ok(path)
}

fn main() {
    //get current cmd args
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().join("bin");
    println!("{}",exec_dir_path.display());
    // 创建一个新的 Command 结构体，并指定要执行的程序名称
    let mut command = Command::new(exec_dir_path.join("diverseupdate.exe")).current_dir(exec_dir_path.as_path()).spawn().unwrap();
    // 启动进程
    let output = command.wait_with_output().unwrap();

    // 打印输出结果
    println!("{}", String::from_utf8_lossy(&output.stdout));
    // println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    let  mut cmd = Command::new(exec_dir_path.join("diverseshot.exe"));
    if args.len() >= 2 {
        //get project_name from args 0
        let arg = format!("--open_project={}", args.get(1).unwrap());
        cmd.arg(arg);
    }
    
    let path = format!("{};{}", exec_dir_path.join("sfm").to_string_lossy(), exec_dir_path.join("torch_dll").to_string_lossy());
    println!("path: {}", path);
    cmd.env("PATH", path);
    cmd.current_dir(exec_dir_path.as_path());
    let child = cmd.spawn().unwrap();
    // child.detach();
    std::mem::forget(child);
    // println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
}
