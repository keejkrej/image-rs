import { Fragment } from "react";

import {
  Menubar,
  MenubarContent,
  MenubarItem,
  MenubarMenu,
  MenubarSeparator,
  MenubarShortcut,
  MenubarSub,
  MenubarSubContent,
  MenubarSubTrigger,
  MenubarTrigger,
} from "@/components/ui/menubar";
import type { CommandDispatchEvent, MenuItem, TopLevelMenu } from "@/types/menu";

type MenuShellProps = {
  menus: TopLevelMenu[];
  onDispatch: (event: CommandDispatchEvent) => void;
  implementedCommands: Set<string>;
};

function dispatchMenuItem(
  id: string,
  label: string,
  implementedCommands: Set<string>,
  onDispatch: (event: CommandDispatchEvent) => void
) {
  onDispatch({
    commandId: id,
    label,
    implemented: implementedCommands.has(id),
    source: "menu",
  });
}

function renderItem(
  item: MenuItem,
  key: string,
  implementedCommands: Set<string>,
  onDispatch: (event: CommandDispatchEvent) => void
) {
  if (item.type === "separator") {
    return <MenubarSeparator key={key} />;
  }

  if (item.type === "submenu") {
    return (
      <MenubarSub key={key}>
        <MenubarSubTrigger disabled={item.enabled === false}>{item.label}</MenubarSubTrigger>
        <MenubarSubContent>
          {item.items.map((child, index) => (
            <Fragment
              key={
                child.type === "separator"
                  ? `${key}.sep.${index}`
                  : `${key}.${child.id}`
              }
            >
              {renderItem(
                child,
                child.type === "separator"
                  ? `${key}.sep.${index}`
                  : `${key}.${child.id}`,
                implementedCommands,
                onDispatch
              )}
            </Fragment>
          ))}
        </MenubarSubContent>
      </MenubarSub>
    );
  }

  return (
    <MenubarItem
      key={key}
      disabled={item.enabled === false}
      onSelect={() => dispatchMenuItem(item.id, item.label, implementedCommands, onDispatch)}
    >
      {item.label}
      {item.shortcut ? <MenubarShortcut>{item.shortcut}</MenubarShortcut> : null}
    </MenubarItem>
  );
}

export function MenuShell({ menus, onDispatch, implementedCommands }: MenuShellProps) {
  return (
    <Menubar>
      {menus.map((menu) => (
        <MenubarMenu key={menu.id}>
          <MenubarTrigger>{menu.label}</MenubarTrigger>
          <MenubarContent>
            {menu.items.map((item, index) => (
              <Fragment
                key={
                  item.type === "separator"
                    ? `${menu.id}.sep.${index}`
                    : `${menu.id}.${item.id}`
                }
              >
                {renderItem(
                  item,
                  item.type === "separator"
                    ? `${menu.id}.sep.${index}`
                    : `${menu.id}.${item.id}`,
                  implementedCommands,
                  onDispatch
                )}
              </Fragment>
            ))}
          </MenubarContent>
        </MenubarMenu>
      ))}
    </Menubar>
  );
}
