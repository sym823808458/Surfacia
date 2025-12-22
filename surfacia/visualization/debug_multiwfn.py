import os
import sys
import subprocess
from datetime import datetime

class BatchMultiwfnGenerator:
    """
    一个独立的、用于批量处理整个文件夹中.fchk文件的类。
    它精确地复制了经过验证的成功脚本的核心逻辑：
    1. 使用 os.chdir() 切换到文件所在目录。
    2. 使用 subprocess.run() 并通过 input 参数传递指令。
    """

    @staticmethod
    def _create_content(surface_type, sample_name):
        """(内部方法) 根据类型生成对应的Multiwfn输入指令"""
        if surface_type == 'ESP':
            return "12\n1\n1\n0.01\n0\n6\n-1\n0\n"
        elif surface_type == 'ALIE':
            return "12\n2\n2\n1\n1\n0.01\n0\n6\n-1\n0\n"
        elif surface_type == 'LEAE':
            xyz_pdb_file = f"{sample_name}_xyz.pdb"
            return f"12\n2\n-4\n1\n1\n0.01\n0\n6\n-1\n0\n"
        return None

    @staticmethod
    def _generate_xyz_pdb_prerequisite(fchk_filename_in_cwd):
        """
        (内部方法) 为LEAE计算生成必需的 _xyz.pdb 文件。
        此方法假定当前工作目录已经是fchk文件所在的目录。
        """
        sample_name = os.path.splitext(fchk_filename_in_cwd)[0]
        output_xyz_pdb = f"{sample_name}_xyz.pdb"

        if os.path.exists(output_xyz_pdb):
            print(f"    - 依赖文件 '{output_xyz_pdb}' 已存在，跳过。")
            return True
        
        print(f"    - 正在为LEAE生成依赖文件: {output_xyz_pdb}...")
        content = "100\n2\n1\n"
        command = ["Multiwfn_noGUI", fchk_filename_in_cwd]

        if os.path.exists('vtx.pdb'):
            os.remove('vtx.pdb')

        result = subprocess.run(
            command, input=content, text=True, capture_output=True
        )

        if os.path.exists('vtx.pdb'):
            os.rename('vtx.pdb', output_xyz_pdb)
            print(f"      ✅ 成功创建依赖文件。")
            return True
        else:
            print(f"      ❌ 创建依赖文件失败。返回码: {result.returncode}")
            print(f"      Stderr: {result.stderr.strip()}")
            return False

    @staticmethod
    def process_single_fchk_file(fchk_file_path):
        """
        对单个.fchk文件，处理所有三种表面类型 (ESP, LEAE, ALIE)。
        这是本类的主要工作函数。
        """
        print("-" * 70)
        print(f"▶️ 开始处理文件: {os.path.basename(fchk_file_path)}")

        try:
            fchk_file_abs = os.path.abspath(fchk_file_path)
            input_dir = os.path.dirname(fchk_file_abs)
            fchk_filename = os.path.basename(fchk_file_abs)
            sample_name = os.path.splitext(fchk_filename)[0]
        except Exception as e:
            print(f"  ❌ 路径计算时出错: {e}")
            return

        original_dir = os.getcwd()
        try:
            # 【核心步骤】切换到文件所在目录
            os.chdir(input_dir)

            # 1. 首先，为LEAE准备好依赖文件
            leae_prereq_ok = BatchMultiwfnGenerator._generate_xyz_pdb_prerequisite(fchk_filename)

            # 2. 依次处理每一种表面类型
            for surface_type in ['ESP', 'ALIE', 'LEAE']:
                print(f"  - 正在计算 {surface_type} 表面...")

                # 如果是LEAE且依赖文件失败，则跳过
                if surface_type == 'LEAE' and not leae_prereq_ok:
                    print(f"    ❌ 因依赖文件生成失败，跳过LEAE计算。")
                    continue

                output_pdb_file = f"{sample_name}_{surface_type}.pdb"
                if os.path.exists(output_pdb_file):
                    print(f"    ✅ 文件 '{output_pdb_file}' 已存在，跳过。")
                    continue

                content = BatchMultiwfnGenerator._create_content(surface_type, sample_name)
                command = ["Multiwfn_noGUI", fchk_filename]
                
                if os.path.exists('vtx.pdb'):
                    os.remove('vtx.pdb')

                result = subprocess.run(
                    command, input=content, text=True, capture_output=True
                )

                if os.path.exists('vtx.pdb'):
                    os.rename('vtx.pdb', output_pdb_file)
                    print(f"    ✅ 成功生成: {output_pdb_file}")
                else:
                    print(f"    ❌ 生成失败！返回码: {result.returncode}")
                    if result.stderr:
                        print(f"    Stderr: {result.stderr.strip()}")

        except Exception as e:
            print(f"  ❌ 处理文件时发生意外的Python错误: {e}")
        finally:
            # 【核心步骤】无论如何，都要切换回原始目录
            os.chdir(original_dir)


if __name__ == "__main__":
    # --- 使用说明 ---
    # 在终端中运行此脚本，并提供一个包含.fchk文件的文件夹路径作为参数。
    #
    # 示例:
    # python process_folder.py /path/to/my/fchk_files/
    # python process_folder.py .  (如果脚本和fchk文件在同一目录)
    # ----------------

    if len(sys.argv) != 2:
        print("\n错误: 请提供一个文件夹路径作为参数。")
        print("用法: python process_folder.py <folder_path>")
        sys.exit(1)

    target_folder = sys.argv[1]

    if not os.path.isdir(target_folder):
        print(f"\n错误: 提供的路径不是一个有效的文件夹 -> '{target_folder}'")
        sys.exit(1)

    print(f"🚀 开始批量处理文件夹: {os.path.abspath(target_folder)}")
    start_time = datetime.now()

    # 查找所有.fchk文件 (不区分大小写)
    fchk_files = [f for f in os.listdir(target_folder) if f.lower().endswith('.fchk')]

    if not fchk_files:
        print("\n在该文件夹中未找到任何 .fchk 文件。")
        sys.exit(0)

    print(f"🔍 找到了 {len(fchk_files)} 个 .fchk 文件，准备开始处理...")

    # 对找到的每一个fchk文件，调用处理函数
    for i, filename in enumerate(fchk_files):
        full_path = os.path.join(target_folder, filename)
        BatchMultiwfnGenerator.process_single_fchk_file(full_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print("-" * 70)
    print(f"🎉 全部处理完成！")
    print(f"总计用时: {duration}")
    print("-" * 70)