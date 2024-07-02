use std::process::Command;

fn main() {
    // 创建一个新的 Command 结构体，并指定要执行的程序名称
    let mut command = Command::new("diverseupdate.exe").spawn().unwrap();
    // 启动进程
    let output = command.wait_with_output().unwrap();

    // 打印输出结果
    println!("{}", String::from_utf8_lossy(&output.stdout));
    // println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    let  mut child = Command::new("diverseshot.exe").spawn().unwrap();
    // child.detach();
    std::mem::forget(child);
    // println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
}
