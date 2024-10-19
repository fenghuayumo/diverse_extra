use std::process::Command;
use std::fs;
fn getExecutablePath() -> std::io::Result<std::path::PathBuf> {
    let path = std::env::current_exe()?;
    Ok(path)
}

fn pre_dll_has_exist()->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll"];
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap();
    let torch_folder = exec_dir_path
    .join("Python/Lib/site-packages/torch/lib");

    for name in global_file_name.iter(){
        let newpath = torch_folder.join(name);
        if !newpath.exists() {
            return false;
        }
    }
    return true;
}
fn main() {
    //get current cmd args
    let args: Vec<String> = std::env::args().collect();
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().join("bin");
    //read the json file to get the dependencies torch url
    let mut torch_url  = String::from("https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip");
    let json_path = exec_path.parent().unwrap().join("Dependencies.json");
    if json_path.exists()  {
        let json_str = std::fs::read_to_string(json_path.clone()).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        let dependencies = json["dependencies"].as_array().unwrap();
        for dep in dependencies.iter() {
            //whether dep["name"].as_str() == "torch"
            if dep["name"].as_str().unwrap() == "torch" {
                torch_url = dep["url"].as_str().unwrap().to_string();
            }
        }
    }
  
    //check update
    let mut need_install_dep = !pre_dll_has_exist();
    if exec_dir_path.join("AutoUpdateInCSharp.exe").exists() {
        let mut command = Command::new(exec_dir_path.join("AutoUpdateInCSharp.exe"));
        let arg = format!("Update");
        command.arg(arg);
        command.current_dir(exec_dir_path.as_path());
        let child = command.spawn().unwrap();
        let output = child.wait_with_output().unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
        let version_file = exec_dir_path.join("Version.json");
        if version_file.exists() {
            let version_str = fs::read_to_string(version_file).unwrap();
            let version_json: serde_json::Value = serde_json::from_str(&version_str).unwrap();
            let version_dep = version_json["dependencies"].as_str().unwrap().to_string();
            println!("version_dep: {}", version_dep);
            if version_dep != torch_url {
                torch_url = version_dep;
                need_install_dep = true;
            }
            //write the version to the dependencies.json file
            let json_str = fs::read_to_string(json_path.clone()).unwrap();
            let mut json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
            json["dependencies"][0]["url"] = serde_json::Value::String(torch_url.clone());
            let new_json_str = serde_json::to_string_pretty(&json).unwrap();
            fs::write(json_path, new_json_str).unwrap();
        }
    }
    //install dependencies
    while  need_install_dep {
        let mut command = Command::new(exec_dir_path.join("diverseupdate.exe"));
        let arg = format!("{}", torch_url);
        command.arg(arg);
        command.current_dir(exec_dir_path.as_path());
        let child = command.spawn().unwrap();
        let output = child.wait_with_output().unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
        need_install_dep = !pre_dll_has_exist();
    }

    let  mut cmd = Command::new(exec_dir_path.join("diverseshot.exe"));
    if args.len() >= 2 {
        //get project_name from args 0
        let arg = format!("--open_project={}", args.get(1).unwrap());
        cmd.arg(arg);
    }
    let pypath = exec_path.parent().unwrap().join("Python");
    let path = format!("{};{}", exec_dir_path.join("sfm").to_string_lossy(), pypath.join("Lib/site-packages/torch/lib").to_string_lossy());
    println!("path: {}", path);
    cmd.env("PATH", path);
    cmd.current_dir(exec_dir_path.as_path());
    let child = cmd.spawn().unwrap();
    // child.detach();
    std::mem::forget(child);
}
