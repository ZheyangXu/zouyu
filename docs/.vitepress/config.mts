import { defineConfig } from "vitepress";

export default defineConfig({
  title: "邹吾",
  description: "林氏国 ，有珍兽，大若虎，五采毕具，尾长于身，名曰驺吾，乘之日行千里。",
  base: "/zouyu/",
  themeConfig: {
    nav: [
      { text: "Home", link: "/" },
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
          items: [
            {
              text: "Mujoco",
              link: "/robotics/simulation/mujoco/",
            },
          ],
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
    math: true,
    lineNumbers: true,
  },
});
