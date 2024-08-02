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
    progress: & mut f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if pre_dll_has_exist() {
        println!("The prerequisite  package has existed, no need to download again");
        return Ok(());
    }
    // 创建一个 Reqwest 客户端
    let client = Client::new();

    // 下载文件
    let res = client.get(url).send().await?;

    // 获取文件大小
    let total_size = res.content_length().ok_or("Failed to get content length")?;

    // 设置进度条
    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading prerequisite  package : torch"));

    // 创建一个文件用于保存下载的文件
    let mut file = File::create(output_path)?;

    // 读取响应流并写入文件
    let mut stream = res.bytes_stream();
    let mut downloaded: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        let new = downloaded + (chunk.len() as u64);
        downloaded = new;
        pb.set_position(new);
        *progress = new as f32 / total_size as f32;
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

#[derive(PartialEq)]
enum UpdateState {
    Idle,
    InProgress,
    Finished,
}

impl Default for UpdateState {
    fn default() -> Self {
        UpdateState::Idle
    }
}

#[derive(Default)]
struct DiverseUpdateApp {
    update_state: UpdateState,
    progress: f32,
}

async fn async_run(progress : & mut MutexGuard<'_, f32>) -> Result<(), Box<dyn std::error::Error>> {
    // let url = "https://github.com/zhengzhang01/Pixel-GS/archive/refs/heads/main.zip";
    let url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip";
    let output_path = "temp.zip";

    download_and_extract(url, output_path, progress).await?;

    Ok(())
}

impl eframe::App for DiverseUpdateApp {
    fn update(&mut self, ctx: &Context, frame: &mut eframe::Frame) {
        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                
            });
        });

        if self.update_state == UpdateState::InProgress {
            self.progress += 0.01;
            if self.progress >= 1.0 {
                self.update_state = UpdateState::Finished;
                self.progress = 0.0;
            }
        }

        // 显示更新进度条
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.update_state == UpdateState::InProgress {
                ui.add(egui::ProgressBar::new(self.progress));
            }
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
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0])
        .with_minimize_button(false).with_maximize_button(false).with_resizable(false),
        ..Default::default()
    };
    let mut app = Box::<DiverseUpdateApp>::default();
    let  progress = Arc::new(Mutex::new(app.progress));
    tokio::spawn(async move {
        let mut progress = progress.lock().unwrap();
        async_run(&mut progress).await;
    });
    eframe::run_native(
        "Download",
        native_options,
        Box::new(|cc|  Ok(app)),
    )
    .unwrap();
    Ok(())
}
// struct MyApp {
//     name: String,
//     age: u32,
// }

// impl Default for MyApp {
//     fn default() -> Self {
//         Self {
//             name: "Arthur".to_owned(),
//             age: 42,
//         }
//     }
// }

// impl eframe::App for MyApp {
//     fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
//         egui::CentralPanel::default().show(ctx, |ui| {
//             ui.heading("My egui Application");
//             ui.horizontal(|ui| {
//                 let name_label = ui.label("Your name: ");
//                 ui.text_edit_singleline(&mut self.name)
//                     .labelled_by(name_label.id);
//             });
//             ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
//             if ui.button("Increment").clicked() {
//                 self.age += 1;
//             }
//             ui.label(format!("Hello '{}', age {}", self.name, self.age));

//             // ui.image(egui::include_image!(
//             //     "../../../crates/egui/assets/ferris.png"
//             // ));
//         });
//     }
// }
// fn main() -> eframe::Result {
//     env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
//     let options = eframe::NativeOptions {
//         viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0])
//         .with_minimize_button(false).with_maximize_button(false).with_resizable(false),
//         ..Default::default()
//     };
//     eframe::run_native(
//         "My egui App",
//         options,
//         Box::new(|cc| {
//             // This gives us image support:
//             egui_extras::install_image_loaders(&cc.egui_ctx);

//             Ok(Box::<MyApp>::default())
//         }),
//     )
// }