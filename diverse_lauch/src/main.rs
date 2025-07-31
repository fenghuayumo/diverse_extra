use std::path::Path;
use std::process::Command;
use std::{fs, vec};
use std::collections::HashMap;
fn getExecutablePath() -> std::io::Result<std::path::PathBuf> {
    let path = std::env::current_exe()?;
    Ok(path)
}

fn pre_dll_has_exist()->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll"];
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap();
    let torch_folder = exec_dir_path
    .join("torch/lib");

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
    let current_dir = exec_path.parent().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().join("bin");
    //read the json file to get the dependencies torch url
    let mut torch_url  = String::from("https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.4.1%2Bcu118.zip");
    let write_json_path = exec_dir_path.join("current_dependencies.json");
    let mut exist_dependencies : Vec<serde_json::value::Value> = vec![];
    if write_json_path.exists() {
        let json_str = std::fs::read_to_string(write_json_path.clone()).unwrap();
        let json : Result<serde_json::Value, serde_json::Error>= serde_json::from_str(&json_str);
        if json.is_ok() {
            let json = json.unwrap();
            let deps = json["dependencies"].as_array().unwrap();
            for dep in deps.iter() {
                exist_dependencies.push(dep.clone());
            }
        }
    }
    let json_path = exec_path.parent().unwrap().join("Dependencies.json");
    if json_path.exists()  {
        let json_str = std::fs::read_to_string(json_path.clone()).unwrap();
        let json : Result<serde_json::Value, serde_json::Error>= serde_json::from_str(&json_str);
        if json.is_ok() {
            let json = json.unwrap();
            let dependencies = json["dependencies"].as_array().unwrap();
            for dep in dependencies.iter() {
                let mut need_install_dep = false;
                let outpath = dep["output"].as_str().unwrap();
                need_install_dep = !current_dir.join(outpath).join(dep["name"].as_str().unwrap()).exists();
                if exist_dependencies.iter().any(|edep| edep["name"].as_str().unwrap() == dep["name"].as_str().unwrap() && edep["url"].as_str().unwrap() != dep["url"].as_str().unwrap()) {
                    need_install_dep = true;
                }
                while need_install_dep {
                    let mut command = Command::new(exec_dir_path.join("splatX_download.exe"));
                    command.arg(format!("{}",dep["name"].as_str().unwrap()));
                    command.arg(format!("{}",dep["url"].as_str().unwrap()));
                    command.arg(format!("{}",outpath));
                    command.current_dir(exec_dir_path.as_path());
                    let child = command.spawn().unwrap();
                    let output = child.wait_with_output().unwrap();
                    println!("{}", String::from_utf8_lossy(&output.stdout));
                    need_install_dep = !current_dir.join(outpath).join(dep["name"].as_str().unwrap()).exists();
                }
            
                if !need_install_dep {
                    //update the exist_dependencies with the new dep url, if the dep is not in the exist_dependencies, add it
                    let mut is_exist = false;
                    for edep in exist_dependencies.iter_mut() {
                        if edep["name"].as_str().unwrap() == dep["name"].as_str().unwrap() {
                            edep["url"] = serde_json::Value::String(dep["url"].as_str().unwrap().to_string());
                            is_exist = true;
                        }
                    }
                    if !is_exist {
                        exist_dependencies.push(dep.clone());
                    }
                }
            }
        }
    }
    //write the exist_dependencies to the json file as dependencies
    let mut json = serde_json::Value::Object(serde_json::Map::new());
    json["dependencies"] = serde_json::Value::Array(exist_dependencies);
    let json_str = serde_json::to_string_pretty(&json).unwrap();
    fs::write(write_json_path, json_str).unwrap();
    //check update
    if exec_dir_path.join("AutoUpdate").join("AutoUpdateInCSharp.exe").exists() {
        let mut command = Command::new(exec_dir_path.join("AutoUpdate").join("AutoUpdateInCSharp.exe"));
        let arg = format!("Update");
        command.arg(arg);
        command.current_dir(exec_dir_path.as_path());
        let child = command.spawn().unwrap();
        let output = child.wait_with_output().unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }
    //sleep 2s
    std::thread::sleep(std::time::Duration::from_secs(2));
    //install dependencies
    //convert onnx to engine
    if !current_dir.join("models").join("mask_general.engine").exists() {
        println!("convert onnx to engine, first time will cost a few minutes");
        let mut command = Command::new(current_dir.join("TensorRT-10.12.0.36/bin/trtexec.exe"));
        let onnx_path = current_dir.join("models").join("mask_general.onnx");
        let engine_path = current_dir.join("models").join("mask_general.engine");
        let arg = format!("--onnx={}", onnx_path.to_str().unwrap());
        command.arg(arg);
        let args1 = format!("--saveEngine={}", engine_path.to_str().unwrap());
        command.arg(args1);
        if json_path.exists()  {
            let json_str = std::fs::read_to_string(json_path.clone()).unwrap();
            let json : Result<serde_json::Value, serde_json::Error>= serde_json::from_str(&json_str);
            if json.is_ok() {
                let json = json.unwrap();
                let env_path = json["env_path"].as_str().unwrap();
                // split the env_path by ;
                let paths = env_path.split(";");
                let mut env_path_str = String::new();
                for path in paths {
                    let path = current_dir.join(path);
                    env_path_str.push_str(path.to_str().unwrap());
                    env_path_str.push(';');
                }
                command.env("PATH", env_path_str);
            }
        }
        command.current_dir(current_dir);
        let child = command.spawn().unwrap();
        let output = child.wait_with_output().unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
    }
    //如果参数包含--inputPath, 则创建splatx-cli 进程
    let params = parse_args(&args);
    
    // 处理 --help 参数
    if params.contains_key("--help") {
        // print the help message
        println!("Usage: splatx-cli [OPTIONS]");
        println!("Options:");
        println!("  -h,--help                   Print this help message and exit");
        println!("  --inputPath TEXT            set data set path");
        println!("  --outputPath TEXT [../out_put/iteration]");
        println!("  --exportMesh BOOLEAN [0]    whether enable normal loss, 0: no, 1: yes");
        println!("  --modelType INT [0]         set trained model type, 0: Splat3D , 1: Splat2D");
        println!("  --densifyStrategy INT [1]   set denisfy strategy, 0: SplatADC, 1: SplatMCMC");
        println!("  --maxIteration INT [30000]  set max iterations");
        println!("  --load_itr INT [-1]         loaditerations");
        println!("  --mipAntiliased BOOLEAN [0]  whether enable mipAntiliased training, 0: no, 1: yes");
        println!("  --packLevel INT [0]         set pack level which can optimize memory usage, 0: no, 1: yes");
        return;
    }
    
    if params.contains_key("--inputPath") {
        let mut command = Command::new(exec_dir_path.join("splatX-cli.exe"));
        
        // 从HashMap中获取参数值
        if let Some(input_path) = params.get("--inputPath") {
            command.arg(format!("--inputPath={}", input_path));
        }
        
        if let Some(output_path) = params.get("--outputPath") {
            command.arg(format!("--outputPath={}", output_path));
        }
        
        if let Some(export_mesh) = params.get("--exportMesh") {
            command.arg(format!("--exportMesh={}", export_mesh));
        }
        
        if let Some(model_type) = params.get("--modelType") {
            command.arg(format!("--modelType={}", model_type));
        }
        
        if let Some(densify_strategy) = params.get("--densifyStrategy") {
            command.arg(format!("--densifyStrategy={}", densify_strategy));
        }
                 if let Some(anti_alias) = params.get("--antiAlias") {
             command.arg(format!("--mipAntiliased={}", anti_alias));
         }
         
         // 添加其他可选参数
         if let Some(max_iteration) = params.get("--maxIteration") {
             command.arg(format!("--maxIteration={}", max_iteration));
         }
         
         if let Some(load_itr) = params.get("--load_itr") {
             command.arg(format!("--load_itr={}", load_itr));
         }
         
         if let Some(pack_level) = params.get("--packLevel") {
             command.arg(format!("--packLevel={}", pack_level));
         } else {
             command.arg("--packLevel=1");
         }
        if json_path.exists()  {
            let json_str = std::fs::read_to_string(json_path.clone()).unwrap();
            let json : Result<serde_json::Value, serde_json::Error>= serde_json::from_str(&json_str);
            if json.is_ok() {
                let json = json.unwrap();
                let env_path = json["env_path"].as_str().unwrap();
                // split the env_path by ;
                let paths = env_path.split(";");
                let mut env_path_str = String::new();
                for path in paths {
                    let path = current_dir.join(path);
                    env_path_str.push_str(path.to_str().unwrap());
                    env_path_str.push(';');
                }
                command.env("PATH", env_path_str);
            }
        }
        command.current_dir(exec_dir_path.as_path());
        let child = command.spawn().unwrap();
        let output = child.wait_with_output().unwrap();
        println!("{}", String::from_utf8_lossy(&output.stdout));
        return;
    }
    let  mut cmd = Command::new(exec_dir_path.join("SplatX.exe"));
    if args.len() >= 2 {
        //get project_name from args 0
        let arg = format!("--open_project={}", args.get(1).unwrap());
        cmd.arg(arg);
    }
    if json_path.exists()  {
        let json_str = std::fs::read_to_string(json_path.clone()).unwrap();
        let json : Result<serde_json::Value, serde_json::Error>= serde_json::from_str(&json_str);
        if json.is_ok() {
            let json = json.unwrap();
            let env_path = json["env_path"].as_str().unwrap();
            // split the env_path by ;
            let paths = env_path.split(";");
            let mut env_path_str = String::new();
            for path in paths {
                let path = current_dir.join(path);
                env_path_str.push_str(path.to_str().unwrap());
                env_path_str.push(';');
            }
            cmd.env("PATH", env_path_str);
        }
    }
    cmd.current_dir(exec_dir_path.as_path());
    let child = cmd.spawn().unwrap();
    // child.detach();
    std::mem::forget(child);
}

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut params = HashMap::new();
    
    for arg in args.iter().skip(1) { // 跳过程序名
        if arg.starts_with("--") {
            if let Some(equal_pos) = arg.find('=') {
                let key = arg[..equal_pos].to_string();
                let value = arg[equal_pos + 1..].to_string();
                params.insert(key, value);
            } else {
                // 处理没有值的标志参数
                params.insert(arg.clone(), "true".to_string());
            }
        }
    }
    
    params
}
