#import <Cocoa/Cocoa.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
        [NSApp activateIgnoringOtherApps:YES];

        NSOpenPanel *panel = [NSOpenPanel openPanel];
        panel.canChooseFiles = YES;
        panel.canChooseDirectories = YES;
        panel.allowsMultipleSelection = YES;
        panel.canCreateDirectories = YES;
        panel.title = @"Select Dataset (File or Folder)";
        panel.prompt = @"Select";

        // Argument 1: mode ("both", "folder", "file")
        if (argc > 1) {
            NSString *mode = [NSString stringWithUTF8String:argv[1]];
            if ([mode isEqualToString:@"folder"]) {
                panel.canChooseFiles = NO;
                panel.canChooseDirectories = YES;
                panel.allowsMultipleSelection = NO;
                panel.title = @"Select Destination Folder";
            } else if ([mode isEqualToString:@"file"]) {
                panel.canChooseFiles = YES;
                panel.canChooseDirectories = NO;
                panel.allowsMultipleSelection = YES;
                panel.title = @"Select Dataset File(s)";
            } else {
                panel.canChooseFiles = YES;
                panel.canChooseDirectories = YES;
                panel.allowsMultipleSelection = YES;
                panel.title = @"Select Dataset (File, Files, or Folder)";
            }
        }

        // Argument 2: initial directory
        if (argc > 2) {
            NSString *dirPath = [NSString stringWithUTF8String:argv[2]];
            BOOL isDir = NO;
            if ([[NSFileManager defaultManager] fileExistsAtPath:dirPath isDirectory:&isDir]) {
                if (!isDir) {
                    dirPath = [dirPath stringByDeletingLastPathComponent];
                }
                panel.directoryURL = [NSURL fileURLWithPath:dirPath];
            }
        }

        // Argument 3: custom title
        if (argc > 3) {
            panel.title = [NSString stringWithUTF8String:argv[3]];
        }

        [panel makeKeyAndOrderFront:nil];
        [panel setLevel:NSFloatingWindowLevel];

        NSModalResponse response = [panel runModal];
        if (response == NSModalResponseOK) {
            for (NSURL *selectedURL in [panel URLs]) {
                if (selectedURL) {
                    printf("%s\n", [[selectedURL path] UTF8String]);
                }
            }
            fflush(stdout);
        }
    }
    return 0;
}
