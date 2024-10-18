// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![allow(rustdoc::missing_crate_level_docs)] // it's an example
use eframe::egui::mutex::MutexGuard;
use egui::{Button, Context, Label, TopBottomPanel, Ui};
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
// use eframe::{App, Frame};
use eframe::egui::{self, text};
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
    let global_file_name = vec![
        "torch.dll",
        "torch_cpu.dll",
        "torch_cuda.dll",
        "c10_cuda.dll",
        "c10.dll",
        "cudart64_110.dll",
        "uv.dll",
        "cudnn_ops_infer64_8.dll",
        "cudnn_cnn_infer64_8.dll",
        "asmjit.dll",
        "zlibwapi.dll",
        "nvToolsExt64_1.dll",
        "nvfuser_codegen.dll",
        "cudnn64_8.dll",
        "fbgemm.dll",
        "fbjni.dll",
        "libiomp5md.dll",
        "libiompstubs5md.dll",
        "cublas64_11.dll",
        "cublasLt64_11.dll",
        "cudnn64_8.dll",
        "cufft64_10.dll",
        "cufftw64_10.dll",
    ];
    let exec_path = getExecutablePath().unwrap();
    let exec_dir_path = exec_path.parent().unwrap().parent().unwrap();
    if exec_dir_path
        .join("Python/Lib/site-packages/torch/lib")
        .exists()
    {
        for name in global_file_name.iter() {
            let newpath = exec_dir_path
                .join("Python/Lib/site-packages/torch/lib")
                .join(name);
            println!("Check file: {:?}", newpath);
            if !newpath.exists() {
                return false;
            }
        }
    }else{
        return false;
    }
    return true;
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
        *(progress.lock().unwrap()) = new as f32 / total_size as f32;
    }
    pb.finish_with_message(format!("Downloaded torch to {}", output_path));

    println!("Unzip file....");
    *status.lock().unwrap() = DownloadStatus::Unziping;

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
        let exec_path = getExecutablePath().unwrap();
        let exec_dir_path = exec_path.parent().unwrap().parent().unwrap();
        let new_file = exec_dir_path
            .join("Python/Lib/site-packages/torch/lib")
            .join(file_name);
        let new_path = exec_dir_path.join("Python/Lib/site-packages/torch/lib");
        println!("Move {:?} to {:?}", path, new_file);
        if !new_path.exists() {
            fs::create_dir(new_path.as_path())?;
        }
        if is_torch_pre_dll(path.to_str().unwrap()) {
            fs::rename(path, new_file)?;
        }
        // fs::rename(path, new_path)?;
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

impl eframe::App for DiverseUpdateApp {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        // 显示更新进度条``
        let progess = self.progress.lock().unwrap();
        // set progress bar on windows center
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx().set_visuals(egui::Visuals::dark());
            let image = egui::Image::new(egui::include_image!("../assets/images/logo.png"))
                .fit_to_exact_size(ui.available_size());
            image.paint_at(ui, ui.available_rect_before_wrap());
            let window_size = ui.available_size();
            ui.label(self.download_status.lock().unwrap().to_string());
            ui.add_space(window_size.y / 2.0 - 30.0);
            ui.add(
                egui::ProgressBar::new(*progess)
                    .desired_height(25.0)
                    .show_percentage()
                    .animate(true),
            );
            if *self.download_status.lock().unwrap() == DownloadStatus::Finished {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // get url from the command line
    let mut url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip";
    let args = std::env::args().collect::<Vec<String>>();
    if args.len() >= 2 {
        url = args[1].as_str();
    }
    if pre_dll_has_exist() {
        println!("The prerequisite  package has existed, no need to download again");
        return Ok(());
    }

    // let native_options = eframe::NativeOptions {
    //     viewport: egui::ViewportBuilder::default()
    //         .with_inner_size([480.0, 160.0])
    //         .with_minimize_button(false)
    //         .with_maximize_button(false)
    //         .with_resizable(false),
    //     ..Default::default()
    // };
    let app = DiverseUpdateApp::new();
    let progress = app.progress.clone();
    let status = app.download_status.clone();
    let url = url.to_string();
    // tokio::spawn(async move {
    //     async_run(url, progress, status).await;
    // });
    // eframe::run_native(
    //     "diverse_download",
    //     native_options,
    //     Box::new(|cc| {
    //         egui_extras::install_image_loaders(&cc.egui_ctx);
    //         Ok(Box::new(app))
    //     }),
    // )
    // .unwrap();

    let output_path = "temp.zip";
    download_and_extract(url.as_str(), output_path, progress, status).await?;
    Ok(())
}
