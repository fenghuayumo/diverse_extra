// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example
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
use sevenz_rust;
use std::fmt;
use std::sync::{Arc, Mutex};

#[derive(PartialEq, Eq)]
enum DownloadStatus {
    NotStarted,
    Downloading,
    Unziping,
    Finished,
}

impl Default for DownloadStatus {
    fn default() -> Self {
        DownloadStatus::NotStarted
    }
}

impl fmt::Display for DownloadStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadStatus::NotStarted => write!(f, "Not Started"),
            DownloadStatus::Downloading => write!(f, "Downloading..."),
            DownloadStatus::Unziping => write!(f, "Unziping..."),
            DownloadStatus::Finished => write!(f, "Finished"),
        }
    }
}

fn getExecutablePath() -> std::io::Result<std::path::PathBuf> {
    let path = std::env::current_exe()?;
    Ok(path)
}

//declare a global filename array
fn is_torch_pre_dll(path: &str) -> bool {
    if path.ends_with("dll") {
        return true;
    }
    return false;
}
fn pre_dll_has_exist() -> bool {
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll"];
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().parent().unwrap();
    if exec_dir_path
        .join("torch/lib")
        .exists()
    {
        for name in global_file_name.iter() {
            let newpath = exec_dir_path
                .join("torch/lib")
                .join(name);
            if !newpath.exists() {
                return false;
            }
        }
    }else{
        return false;
    }
    return true;
}

pub async  fn download_package_and_extract(
    url: &str,
    temp_path: &str,
    extract_path: &str,
    name: &str
)-> Result<(), Box<dyn std::error::Error>>{
    // 如果当前目录下存在extract_path/name 文件，则直接返回OK
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().parent().unwrap();
    if exec_dir_path.join(extract_path).join(name).exists() {
        println!("The package file {} has existed, no need to download again", name);
        return Ok(());
    }
    let client = Client::new();
    let res = client.get(url).send().await?;
    let total_size = res.content_length().ok_or("Failed to get content length")?;
    let pb: Arc<ProgressBar> = Arc::new(ProgressBar::new(total_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading prerequisite  package : {}", name));

    let dir = Path::new(temp_path).parent().unwrap();
    if !dir.exists() {
        fs::create_dir_all(dir)?;
    }
    let mut file = File::create(temp_path)?;
    let mut stream = res.bytes_stream();
    let mut downloaded: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        let new = downloaded + (chunk.len() as u64);
        downloaded = new;
        pb.set_position(new);
    }
    pb.finish_with_message(format!("downloaded {} to {}", name,temp_path));
    if Path::new(temp_path).exists() {
        println!("Extracting package file: {}", temp_path);
        // 创建 models 目录
        let models_dir = Path::new(extract_path);
        if !models_dir.exists() {
            fs::create_dir_all(models_dir)?;
        }
        
        // 根据文件扩展名选择解压方法
        if temp_path.ends_with(".7z") {
            // 解压 7z 文件
            sevenz_rust::decompress_file(
                temp_path,
                models_dir
            )?;
        } else if temp_path.ends_with(".zip") {
            // 解压 zip 文件
            let zip_file = File::open(temp_path)?;
            let mut archive = ZipArchive::new(zip_file)?;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                let outpath = models_dir.join(file.sanitized_name());
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
        }
        
        println!("extraction completed");
        // 删除temp_path 文件
        fs::remove_file(temp_path)?;
    }
    Ok(())
}

pub async fn download_and_extract(
    url: &str,
    output_path: &str,
    progress: Arc<Mutex<f32>>,
    status: Arc<Mutex<DownloadStatus>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if pre_dll_has_exist() {
        println!("The prerequisite  package has existed, no need to download again");
        return Ok(());
    }
    let client = Client::new();
    let res = client.get(url).send().await?;
    let total_size = res.content_length().ok_or("Failed to get content length")?;

    *status.lock().unwrap() = DownloadStatus::Downloading;
    let pb: Arc<ProgressBar> = Arc::new(ProgressBar::new(total_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
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
        *(progress.lock().unwrap()) = new as f32 / total_size as f32;
    }
    pb.finish_with_message(format!("Downloaded torch to {}", output_path));

    println!("Extracting file....");
    *status.lock().unwrap() = DownloadStatus::Unziping;

    // 根据文件扩展名选择解压方法
    if output_path.ends_with(".7z") {
        // 解压 7z 文件
        sevenz_rust::decompress_file(
            output_path,
            Path::new(".")
        )?;
    } else if output_path.ends_with(".zip") {
        // 解压 zip 文件
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
    }
    println!("Extraction completed");
    //move the extracted files to the current directory
    let torch_dir = Path::new("libtorch/lib");
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().parent().unwrap();
    let new_path = exec_dir_path.join("torch/lib");
    if !new_path.exists() {
        fs::create_dir_all(new_path.as_path())?;
    }
    let current_path_lib = exec_path.parent().unwrap();
    // let output_path = output_path.parent().unwrap();
    for entry in fs::read_dir(torch_dir)? {
        let entry = entry?;
        let path = entry.path();
        let file_name = path.file_name().unwrap();
        let file_name = file_name.to_str().unwrap();
        let new_file = exec_dir_path
            .join("torch/lib")
            .join(file_name);

        if is_torch_pre_dll(path.to_str().unwrap()) {
            fs::rename(path, new_file)?;
        }
       
        let dll_file = current_path_lib
            .join(entry.path().file_name().unwrap());
        if dll_file.exists() {
            fs::remove_file(dll_file);
        }
    }
    println!("Move completed");
    // delete the extracted folder
    fs::remove_file(output_path)?;
    fs::remove_dir_all("libtorch")?;
    *status.lock().unwrap() = DownloadStatus::Finished;
    Ok(())
}

#[derive(Default)]
struct DiverseUpdateApp {
    progress: Arc<Mutex<f32>>,
    download_status: Arc<Mutex<DownloadStatus>>,
    image_data: Option<Vec<u8>>,
}

async fn async_run(
    url: String,
    progress: Arc<Mutex<f32>>,
    status: Arc<Mutex<DownloadStatus>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = "temp.zip";

    download_and_extract(url.as_str(), output_path, progress, status).await?;
    Ok(())
}

impl DiverseUpdateApp {
    fn new() -> Self {
        DiverseUpdateApp {
            progress: Arc::new(Mutex::new(0.0)), // 初始化 progress 为 0.0
            download_status: Arc::new(Mutex::new(DownloadStatus::NotStarted)),
            image_data: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // get url from the command line
    let mut url = "https://github.com/fenghuayumo/SplatX/releases/download/dep01/mask_general.7z";
    let args = std::env::args().collect::<Vec<String>>();
    let mut name = "mask_general.engine";
    let mut output_path = "models";
    if args.len() >= 2 {
        name = args[1].as_str();
        url = args[2].as_str();
        output_path = args[3].as_str();
    }

    let app = DiverseUpdateApp::new();
    let progress = app.progress.clone();
    let status = app.download_status.clone();
    let url = url.to_string();
    // let output_path = "temp.zip";
    // 根据输入的url 地址确定 temp 压缩包后缀名
    let mut temp_path = "temp.zip";
    if url.contains(".7z") {
        temp_path = "temp.7z";
    }else if url.contains(".zip") {
        temp_path = "temp.zip";
    }
    if name == "torch" {
        if pre_dll_has_exist() {
            println!("The prerequisite  package has existed, no need to download again");
            return Ok(());
        }
        download_and_extract(url.as_str(), temp_path, progress, status).await?;
    }else{
        download_package_and_extract(url.as_str(), temp_path, output_path,name).await?;
    }
    Ok(())
}
