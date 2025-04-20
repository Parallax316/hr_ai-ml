"use client"
import { usePathname } from "next/navigation"
import { X, Upload } from "lucide-react"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetClose } from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { PolicyUploader } from "@/components/policy-uploader"
import { DataUploader } from "@/components/data-uploader"

interface AppSidebarProps {
  open: boolean
  setOpen: (open: boolean) => void
}

export function AppSidebar({ open, setOpen }: AppSidebarProps) {
  const pathname = usePathname()

  return (
    <>
      {/* Desktop sidebar */}
      <div className="hidden w-64 border-r md:block bg-background/60 backdrop-blur-sm">
        <div className="flex items-center gap-2 p-4 border-b">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
            <Upload className="w-4 h-4 text-primary" />
          </div>
          <div className="font-semibold">Document Upload</div>
        </div>
        <div className="p-4">
          {pathname === "/" && <PolicyUploader />}
          {pathname === "/data-analysis" && (
            <>
              <h3 className="mb-2 text-sm font-medium">Upload Employee Data</h3>
              <DataUploader />
            </>
          )}
        </div>
      </div>

      {/* Mobile sidebar */}
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="left" className="w-[280px] p-0">
          <SheetHeader className="p-4 border-b">
            <div className="flex items-center justify-between">
              <SheetTitle className="flex items-center gap-2">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10">
                  <Upload className="w-4 h-4 text-primary" />
                </div>
                <span>Document Upload</span>
              </SheetTitle>
              <SheetClose asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <X className="h-4 w-4" />
                  <span className="sr-only">Close</span>
                </Button>
              </SheetClose>
            </div>
          </SheetHeader>
          <div className="p-4">
            {pathname === "/" && <PolicyUploader />}
            {pathname === "/data-analysis" && (
              <>
                <h3 className="mb-2 text-sm font-medium">Upload Employee Data</h3>
                <DataUploader />
              </>
            )}
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}
