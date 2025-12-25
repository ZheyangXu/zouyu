import { defineConfig } from "vitepress";
import mathjax3 from "markdown-it-mathjax3";

export default defineConfig({
  title: "邹吾",
  description:
    "林氏国 ，有珍兽，大若虎，五采毕具，尾长于身，名曰驺吾，乘之日行千里。",
  base: "/zouyu/",
  themeConfig: {
    nav: [
      {
        text: "邹吾",
        link: "/",
      },
      {
        text: "Projects",
        link: "/projects/",
      },
      {
        text: "机器人学",
        link: "/robotics/",
      },
      {
        text: "控制理论",
        link: "/control-theory/",
      },
      {
        text: "机器人开发",
        // link: "/robotics-development/",
        items: [
          {
            text: "ROS2",
            link: "/robotics-development/ros2/",
          },
          {
            text: "Mujoco",
            link: "/robotics-development/mujoco/",
          },
          {
            text: "Isaac Lab",
            link: "/robotics-development/isaac/",
          },
        ],
      },
      {
        text: "BLOG",
        link: "/blog/",
      },
    ],

    sidebar: {
      "/robotics/": [
        {
          text: "机器人动力学",
          link: "/robotics/dynamics/",
          items: [],
        },
        {
          text: "机器人建模与仿真",
          link: "/robotics/simulation/",
        },
      ],
      "/control-theory/": [
        {
          text: "模型预测控制",
          link: "/control-theory/mpc",
        },
      ],
      "/robotics-development/ros2": [
        {
          text: "ROS2",
          link: "/robotics-development/ros2/",
          items: [
            {
              text: "基础概念",
              link: "/robotics-development/ros2/basic-concepts/",
            },
          ],
        },
      ],
      "/robotics-development/mujoco/": [
        {
          text: "Mujoco",
          items: [
            {
              text: "概要",
              link: "/robotics-development/mujoco/overview/",
            },
            {
              text: "计算",
              link: "/robotics-development/mujoco/computation/",
              items: [],
            },
            {
              text: "建模",
              link: "/robotics-development/mujoco/modeling/",
            },
          ],
        },
      ],
      "/robotics-development/isaac/": [
        {
          text: "Isaac Lab",
          link: "/robotics-development/isaac/",
          items: [
            {
              text: "快速开始",
              link: "/robotics-development/isaac/getting-started/",
              items: [
                {
                  text: "环境设计背景",
                  link: "/robotics-development/isaac/getting-started/environment-design-background/",
                },
                {
                  text: "Classes and Configs",
                  link: "/robotics-development/isaac/getting-started/classes-and-configs/",
                },
                {
                  text: "设计环境",
                  link: "/robotics-development/isaac/getting-started/environment-design/",
                },
                {
                  text: "训练 Jetbot",
                  link: "/robotics-development/isaac/getting-started/training-the-jetbot-gt/",
                },
                {
                  text: "探索 RL 问题",
                  link: "/robotics-development/isaac/getting-started/exploring-the-rl-problem/",
                },
              ],
            },
            {
              text: "基础教程",
              link: "/robotics-development/isaac/tutorials/",
              items: [
                {
                  text: "创建一个空仿真",
                  link: "/robotics-development/isaac/tutorials/create-empty/",
                },
                {
                  text: "生成多种 Prims",
                  link: "/robotics-development/isaac/tutorials/spawn_prims/",
                },
                {
                  text: "深入理解应用程序",
                  link: "/robotics-development/isaac/tutorials/launch-app/",
                },
                {
                  text: "添加新的机器人",
                  link: "/robotics-development/isaac/tutorials/add-new-robot/",
                },
                {
                  text: "运行刚体对象仿真",
                  link: "/robotics-development/isaac/tutorials/run-rigid-object/",
                },
                {
                  text: "运行关节对象仿真",
                  link: "/robotics-development/isaac/tutorials/run-articulation/",
                },
                {
                  text: "运行软体对象仿真",
                  link: "/robotics-development/isaac/tutorials/run-deformable-object/",
                },
                {
                  text: "运行夹爪对象仿真",
                  link: "/robotics-development/isaac/tutorials/run-surface-gripper/",
                },
                {
                  text: "创建交互式场景",
                  link: "/robotics-development/isaac/tutorials/create-scene/",
                },
                {
                  text: "创建基于 Manager 的基础环境",
                  link: "/robotics-development/isaac/tutorials/create-manager-base-env/",
                },
                {
                  text: "创建基于 Manager 的强化学习环境",
                  link: "/robotics-development/isaac/tutorials/create-manager-rl-envs/",
                },
                {
                  text: "创建 Direct Workflow 的强化学习环境",
                  link: "/robotics-development/isaac/tutorials/create-direct-rl-env/",
                },
                {
                  text: "在 Gym 中注册强化学习环境",
                  link: "/robotics-development/isaac/tutorials/register-rl-env-gym/",
                },
                {
                  text: "运行强化学习训练",
                  link: "/robotics-development/isaac/tutorials/run-rl-training/",
                },
                {
                  text: "修改现有 Direct RL 环境",
                  link: "/robotics-development/isaac/tutorials/modify-direct-rl-env/",
                },
                {
                  text: "在 USD 环境中进行策略推理",
                  link: "/robotics-development/isaac/tutorials/policy-inference-in-usd/",
                },
                {
                  text: "在机器人上添加传感器",
                  link: "/robotics-development/isaac/tutorials/add-sensors-on-robot/",
                },
                {
                  text: "使用操作空间控制器(OSC)",
                  link: "/robotics-development/isaac/tutorials/run-osc/",
                },
              ],
            },
            {
              text: "How-to 教程",
              link: "/robotics-development/isaac/how-to/",
              items: [
                {
                  text: "导入新资产",
                  link: "/robotics-development/isaac/how-to/import-new-asset/",
                },
                {
                  text: "编写资产配置",
                  link: "/robotics-development/isaac/how-to/write-articulation-cfg/",
                },
                {
                  text: "在仿真中将物理 prim 固定",
                  link: "/robotics-development/isaac/how-to/make-fixed-prim/",
                },
                {
                  text: "生成多个资产",
                  link: "/robotics-development/isaac/how-to/multi-asset-spawning/",
                },
                {
                  text: "保存渲染图像与 3D 重投影",
                  link: "/robotics-development/isaac/how-to/save-camera-output/",
                },
                {
                  text: "估计可运行的相机数量",
                  link: "/robotics-development/isaac/how-to/estimate-how-many-cameras-can-run/",
                },
                {
                  text: "配置渲染设置",
                  link: "/robotics-development/isaac/how-to/configure-rendering/",
                },
                {
                  text: "创建可视化标记",
                  link: "/robotics-development/isaac/how-to/draw-markers/",
                },
                {
                  text: "包装环境",
                  link: "/robotics-development/isaac/how-to/wrap-rl-env/",
                },
                {
                  text: "添加你自己的学习库",
                  link: "/robotics-development/isaac/how-to/add-own-library/",
                },
                {
                  text: "录制仿真动画",
                  link: "/robotics-development/isaac/how-to/record-animation/",
                },
                {
                  text: "在训练期间录制视频片段",
                  link: "/robotics-development/isaac/how-to/record-video/",
                },
                {
                  text: "课程（Curriculum）工具",
                  link: "/robotics-development/isaac/how-to/curriculums/",
                },
              ],
            },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/ZheyangXU/zouyu.git" },
    ],
    footer: {
      message:
        'Released under the <a href="https://github.com/ZheyangXu/zouyu/main/LICENSE">MIT License</a>.',
      copyright:
        'Copyright © 2024-present <a href="https://github.com/ZheyangXu">ZheyangXu</a>',
    },
  },
  markdown: {
    lineNumbers: true,
    config: (md) => {
      md.use(mathjax3, {
        tex: {
          tags: "ams",
          tagformat: {
            number: (n: number) => n.toString(),
          },
        },
      });
    },
  },
});
