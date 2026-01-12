问题1：
    是否升级 Go 到最新版本？ [y/N]（默认 N，使用兼容模式）: 
    [INFO] 使用兼容模式，将在编译前移除 toolchain 指令。
    [INFO] 依赖安装完成。
    [INFO] 开始安装 graftcp 到：/root/.graftcp-antigravity/graftcp
    [INFO] 克隆 graftcp 仓库...
    [INFO] 尝试从 https://ghproxy.net/https://github.com/hmgle/graftcp.git 克隆...
    Cloning into '/root/.graftcp-antigravity/graftcp'...
    fatal: unable to access 'https://ghproxy.net/https://github.com/hmgle/graftcp.git/': HTTP/2 stream 0 was not closed cleanly: PROTOCOL_ERROR (err 1)
    [WARN] 从 https://ghproxy.net/https://github.com/hmgle/graftcp.git 克隆失败 (退出码: 128)
    [INFO] 尝试从 https://github.com/hmgle/graftcp.git 克隆...
    Cloning into '/root/.graftcp-antigravity/graftcp'...
    [INFO] 仓库克隆成功
    [INFO] 为编译临时设置 GOPROXY=https://goproxy.cn,direct 加速 go 依赖下载（仅本次运行生效）。
    [INFO] 兼容模式：移除 /root/.graftcp-antigravity/graftcp 中 go.mod 的 toolchain 指令...
    [INFO]   注：此修改仅影响 graftcp 仓库，不影响您的其他项目
    [INFO]   移除 local/go.mod 中的 toolchain 行
    [INFO] 开始编译 graftcp（日志写入：/root/.graftcp-antigravity/install.log）...
    [WARN] 编译失败，正在分析原因...
    [INFO] 第 2 次尝试编译...（共 2 次）
    [WARN] 编译失败，正在分析原因...

    ❌ graftcp 编译失败（已尝试 2 次）

解决1：
    使用y 安装

# 1. 强制将新版 Go 加入当前环境路径
export PATH=/usr/local/go/bin:$PATH

# 2. 确认版本（必须看到 go1.25.5 才算成功）
go version

# 3. 修复依赖并编译
cd /root/.graftcp-antigravity/graftcp/local
go mod tidy
cd ..
make

# 1. 告诉脚本直接使用刚才编译好的目录，不要再下载了
export GRAFTCP_DIR=/root/.graftcp-antigravity/graftcp

# 2. 回到脚本所在的目录
cd /opt/data/private

# 3. 再次运行脚本
bash antissh.sh


# 配置永久

echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc && source ~/.bashrc
sed -i '1s/^/export PATH=\/usr\/local\/go\/bin:$PATH\n/' ~/.bashrc

echo 'export PATH=$PATH:/root/.graftcp-antigravity/graftcp' >> ~/.bashrc && source ~/.bashrc

source ~/.bashrc