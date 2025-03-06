import { Calculator, Home, Bot, CalendarX2 } from "lucide-react";
import Logo from "../assets/hatem1.png";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
  useSidebar,
} from "@/components/ui/sidebar";
import { NavLink } from "react-router-dom";

// Menu items.
const items = [
  {
    title: "الرئيسية",
    url: "/",
    icon: Home,
  },
  {
    title: "حاتم",
    url: "/chatbot",
    icon: Bot,
  },
  {
    title: "حساب الغيابات المسموحة",
    url: "/calculator",
    icon: CalendarX2,
  },
  {
    title: "حساب المعدل",
    url: "/gpacalc",
    icon: Calculator,
  },
];

export function AppSidebar() {
  const { open } = useSidebar();

  return (
    <div className="">
      <Sidebar side="right" variant="floating" collapsible="icon">
        <SidebarContent>
          <SidebarGroup>
            <div>
              <SidebarGroupLabel cl>
                <img src={Logo} className="w-8" />

                <SidebarGroupLabel cl>
                  حاتم - مساعدك الأكاديمي
                </SidebarGroupLabel>
              </SidebarGroupLabel>
            </div>
            <SidebarGroupContent>
              <SidebarMenu className="mt-2">
                {items.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink to={item.url}>
                        <item.icon />
                        <span>{item.title}</span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
        <SidebarFooter>
          {open ? (
            <span className="font-light text-sm text-gray-500">
              صنع بحب من
              <a
                className="text-cyan-600 font-medium"
                href="https://www.linkedin.com/in/faisal-al-thuwaini/"
                target="_blank"
              >
                {" "}
                فيصل الثويني{" "}
              </a>
            </span>
          ) : (
            ""
          )}
        </SidebarFooter>
      </Sidebar>
    </div>
  );
}
