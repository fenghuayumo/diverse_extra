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

//declare a global filename array
fn is_torch_pre_dll(path: &str)->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll", 
    "uv.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll","asmjit.dll", "zlibwapi.dll", "nvToolsExt64_1.dll", 
    "nvfuser_codegen.dll", "cudnn64_8.dll"];
    //whether the path is belong to the global_file_name
    let file_name = Path::new(path).file_name().unwrap();
    let file_name = file_name.to_str().unwrap();
    for name in global_file_name.iter(){
        if file_name == *name{
            return true;
        }
    }
    return false;
}
fn pre_dll_has_exist()->bool{
    let global_file_name = vec!["torch.dll", "torch_cpu.dll", "torch_cuda.dll", "c10_cuda.dll", "c10.dll", 
    "uv.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll","asmjit.dll", "zlibwapi.dll", "nvToolsExt64_1.dll", 
    "nvfuser_codegen.dll", "cudnn64_8.dll"];
    for name in global_file_name.iter(){
        let path = Path::new(".").join(name);
        if !path.exists(){
            return false;
        }
    }
    return true;
}

pub async fn download_and_extract(
    url: &str,
    output_path: &str,
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
    }

    pb.finish_with_message(format!("Downloaded {} to {}", url, output_path));

    // 解压文件
    // let file = File::open(output_path)?;
    // let mut archive = Archive::new(GzDecoder::new(file));
    // for entry in archive.entries()? {
    //     let mut entry = entry?;
    //     let path = entry.path()?;
    //     let dest = Path::new(output_path).join(path);
    //     if let Some(parent) = dest.parent() {
    //         fs::create_dir_all(parent)?;
    //     }
    //     entry.unpack(dest)?;
    // }
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let url = "https://github.com/zhengzhang01/Pixel-GS/archive/refs/heads/main.zip";
    let url = "https://download.pytorch.org/libtorch/cu118/libtorch-win-shared-with-deps-2.1.2%2Bcu118.zip";
    let output_path = "temp.zip";

    download_and_extract(url, output_path).await?;

    Ok(())
}
