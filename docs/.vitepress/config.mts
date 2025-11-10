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
        link: "/robotics-development/",
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
      "/robotics-development/": [
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
        {
          text: "Mujoco",
          link: "/robotics-development/mujoco/",
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
