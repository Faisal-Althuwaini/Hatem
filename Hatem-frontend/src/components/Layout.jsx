import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";

function Layout({ children }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarTrigger className="mt-2" />
      <main className="h-screen w-full">{children}</main>
    </SidebarProvider>
  );
}

export default Layout;
