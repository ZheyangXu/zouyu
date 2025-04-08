import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "邹吾",
  description: "邹吾",
  base: "/zouyu/",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      {
        text: "ROS2",
        items: [{ text: "基础概念", link: "/ros2/basic-concepts" }],
      },
      {
        text: "机器人动力学",
        items: [
          {text: "基础概念", link: "/robotics/introduction" },
        ]
      }
    ],

    sidebar: {
      "/ros2/": [
        {
          text: "ROS2",
          items: [{ text: "基础概念", link: "/ros2/basic-concepts" }],
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
