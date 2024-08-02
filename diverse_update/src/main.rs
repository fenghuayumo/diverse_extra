use eframe::egui::mutex::MutexGuard;
use flate2::read::GzDecoder;
use futures::stream::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::io::{prelude::*, Bytes};
use std::path::Path;
use std::time::Instant;
use tar::Archive;
use zip::ZipArchive;
use egui::{Context, Ui, Button, Label, TopBottomPanel};
// use eframe::{App, Frame};
use eframe::egui::{self};
use std::sync::{Arc, Mutex};
fn getExecutablePath() -> std::io::Result<std::path::PathBuf> {
    let path = std::env::current_exe()?;
    Ok(path)
}

//declare a global filename array
fn is_torch_pre_dll(path: &str)->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll", "cudart64_110.dll",
    "uv.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll","asmjit.dll", "zlibwapi.dll", "nvToolsExt64_1.dll", 
    "nvfuser_codegen.dll", "cudnn64_8.dll", "fbgemm.dll", "fbjni.dll"];
    
    //whether the path is belong to the global_file_name
    // let exec_path = getExecutablePath().unwrap();
    // let exec_dir_path = exec_path.parent().unwrap();
    // let binding = exec_dir_path.join(path);
    // let file_name = binding.file_name().unwrap().to_str().unwrap();
    // for name in global_file_name.iter(){
    //     if file_name == *name{
    //         return true;
    //     }
    // }
    if path.ends_with("dll"){
        return true;
    }
    return false;
}
fn pre_dll_has_exist()->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll", "cudart64_110.dll",
    "uv.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll","asmjit.dll", "zlibwapi.dll", "nvToolsExt64_1.dll", 
    "nvfuser_codegen.dll", "cudnn64_8.dll", "fbgemm.dll", "fbjni.dll", "libiomp5md.dll", "libiompstubs5md.dll", 
    "cublas64_11.dll", "cublasLt64_11.dll","cudnn64_8.dll","cufft64_10.dll","cufftw64_10.dll"];
    for name in global_file_name.iter(){
        let exec_path = getExecutablePath().unwrap();
        let exec_dir_path = exec_path.parent().unwrap();
        let path = exec_dir_path.join("bin").join("torch_dll").join(name);
        if !path.exists(){
            return false;
        }
    }
    return true;
}

pub async fn download_and_extract(
    url: &str,
    output_path: &str,
    progress: Arc<Mutex<f32>>
) -> Result<(), Box<dyn std::error::Error>> {
    if pre_dll_has_exist() {
        println!("The prerequisite  package has existed, no need to download again");
        return Ok(());
    }
    let client = Client::new();
    let res = client.get(url).send().await?;
    let total_size = res.content_length().ok_or("Failed to get content length")?;

    let pb: Arc<ProgressBar> = Arc::new(ProgressBar::new(total_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading prerequisite  package : torch"));
    
    let mut file = File::create(output_path)?;
    let mut stream = res.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        let new = downloaded + (chunk.len() as u64);
        downloaded = new;
        pb.set_position(new);
        *(progress.lock().unwrap() )= new as f32 / total_size as f32;
    }
    pb.finish_with_message(format!("Downloaded torch to {}", output_path));
    
    println!("Unzip file....");
    let zip_file = File::open(output_path)?;
    let mut archive = ZipArchive::new(zip_file)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = file.sanitized_name();
        if (&*file.name()).ends_with('/') {
            std::fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }
    println!("Unzip completed");
    //move the extracted files to the current directory
    let torch_dir = Path::new("libtorch/lib");
    // let output_path = output_path.parent().unwrap();
    for entry in fs::read_dir(torch_dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_name = path.file_name().unwrap();
        let file_name = file_name.to_str().unwrap();
        if !Path::new("torch_dll").exists() {
            fs::create_dir("torch_dll");
        }
        let new_path = Path::new("torch_dll").join(file_name);
        //if the file is dll, then move it to the current directory
        if is_torch_pre_dll(path.to_str().unwrap()) {
            fs::rename(path, new_path)?;
        }
        // fs::rename(path, new_path)?;
        
    }
    println!("Move completed");
    // delete the extracted folder
    fs::remove_file(output_path)?;
    fs::remove_dir_all("libtorch")?;
    Ok(())
}

// // #[tokio::main]
// // async fn main() -> Result<(), Box<dyn std::error::Error>> {
// //     // let url = "https://github.com/zhengzhang01/Pixel-GS/archive/refs/heads/main.zip";
// //     let url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip";
// //     let output_path = "temp.zip";

// //     download_and_extract(url, output_path).await?;

// //     Ok(())
// // }

#[derive(Default)]
struct DiverseUpdateApp {
    progress: Arc<Mutex<f32>>
}

async fn async_run(progress: Arc<Mutex<f32>>) -> Result<(), Box<dyn std::error::Error>> {
    // let url = "https://github.com/zhengzhang01/Pixel-GS/archive/refs/heads/main.zip";
    let url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip";
    let output_path = "temp.zip";

    download_and_extract(url, output_path, progress).await?;
    Ok(())
}

impl DiverseUpdateApp {
    fn new() -> Self {
        DiverseUpdateApp {
            progress: Arc::new(Mutex::new(0.0)), // 初始化 progress 为 0.0
        }
    }
}

impl eframe::App for DiverseUpdateApp {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        // 显示更新进度条
        let  progess = self.progress.lock().unwrap();
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(egui::ProgressBar::new(*progess).show_percentage().animate(true));
        });

    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if pre_dll_has_exist() {
        println!("The prerequisite  package has existed, no need to download again");
        return  Ok(());
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([600.0, 240.0])
        .with_minimize_button(false).with_maximize_button(false).with_resizable(false),
        ..Default::default()
    };
    let app = DiverseUpdateApp::new();
    let progress = app.progress.clone();
    tokio::spawn(async move {
        async_run(progress).await;
    });
    eframe::run_native(
        "Download",
        native_options,
        Box::new(|_|  Ok(Box::new(app))),
    )
    .unwrap();
    Ok(())
}